from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import functools

import torch
from torch import nn
torch.set_float32_matmul_precision("medium")
from lightning import LightningModule
from torchmetrics import MeanMetric
from edit_distance import SequenceMatcher
import yaml


# ============================================================
#              PS2S TRANSFORMER (LOADED AS LM)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    # standard causal mask for decoder
    mask = torch.full((sz, sz), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    mask.fill_diagonal_(0.0)
    return mask


def generate_limited_future_mask(sz: int, device: torch.device, lookahead: int = 1) -> torch.Tensor:
    # allow attention to current and up to `lookahead` future steps
    i = torch.arange(sz, device=device).unsqueeze(1)
    j = torch.arange(sz, device=device)
    return torch.where(j - i > lookahead, float("-inf"), 0.0)


class Seq2SeqTransformer(nn.Module):
    """
    Same architecture as your original PS2S phoneme seq2seq model,
    used here as a phoneme LM / denoiser.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        bidirectional: bool = True,  # used for encoder mask
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx

        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.proj = nn.Linear(d_model, vocab_size)

    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        src: (B, S) token ids
        returns: memory (B, S, D), src_key_padding_mask (B, S)
        """
        src_key_padding_mask = src == self.pad_idx
        if self.bidirectional:
            src_mask = generate_limited_future_mask(src.size(1), src.device, lookahead=1)
        else:
            src_mask = generate_square_subsequent_mask(src.size(1), src.device)

        src_emb = self.pos_enc(self.src_emb(src))
        memory = self.encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        return memory, src_key_padding_mask

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """
        src: (B, S) noisy tokens
        tgt_in: (B, T) shifted-clean tokens (with <sos> at position 0)
        returns: logits (B, T, V)
        """
        memory, src_key_padding_mask = self.encode(src)
        tgt_key_padding_mask = tgt_in == self.pad_idx
        tgt_mask = generate_square_subsequent_mask(tgt_in.size(1), tgt_in.device)

        tgt_emb = self.pos_enc(self.tgt_emb(tgt_in))
        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.proj(out)


# ============================================================
#                  MAIN LIGHTNING MODULE
# ============================================================

class SpeechModuleCTCWithPS2SLM(LightningModule):
    """
    Modern-style architecture:

      - Acoustic model: GRU-based CTC network (pretrained, loadable from .ckpt)
      - Text prior: PS2S Transformer trained on massive text (phoneme-level)

    Training:
      - CTC loss on GRU logits.
      - Optional LM fine-tuning loss on PS2S using only labels (phoneme text-only).

    Decoding (val/test):
      - If use_shallow_fusion:
          * CTC beam search → N-best sequences (CTC scores)
          * PS2S LM scores each candidate given a fixed noisy anchor
          * fused score = log P_CTC + lm_weight * log P_LM
      - Else if PS2S exists:
          * Greedy CTC → PS2S post-correction.
      - Else:
          * Plain greedy CTC.

    CER is computed on the final decoded sequence.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Any,
        scheduler: Any,
        compile: bool,
        lr_start: float = 0.02,
        lr_end: float = 0.02,
        n_batch: int = 10000,
        n_units: int = 1024,
        n_layers: int = 5,
        n_classes: int = 40,
        n_input_features: int = 256,
        dropout: float = 0.4,
        gaussian_smooth_width: float = 2.0,
        stride_len: int = 4,
        kernel_len: int = 32,
        bidirectional: bool = False,
        l2_decay: float = 1e-5,
        seed: int = 0,
        # GRU checkpoint
        gru_ckpt_path: str = "data/gru512.ckpt",
        # PS2S checkpoint (from your notebook)
        ps2s_model_path: str = "data/phoneme_seq2seq.pt",
        ps2s_max_len: int = 128,
        phoneme_labels: Optional[List[str]] = None,
        # decoding / LM integration
        use_shallow_fusion: bool = False,
        beam_size: int = 8,
        lm_weight: float = 0.0,          # shallow-fusion LM weight (decoding)
        lm_finetune_weight: float = 0.0, # training-time LM fine-tuning weight
    ) -> None:
        super().__init__()

        if net is None:
            raise ValueError("`net` must be provided.")

        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

        # ---------------- GRU LOADING -----------------
        self._load_gru_weights(gru_ckpt_path)

        # ---------------- CTC -----------------
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        # ---------------- PS2S LM -----------------
        self.ps2s_model_path = ps2s_model_path
        self.ps2s_max_len = ps2s_max_len

        self.ps2s_model: Optional[Seq2SeqTransformer] = None
        self.ps2s_meta: Optional[Dict[str, Any]] = None

        # load PS2S as LM (initially frozen)
        self._load_ps2s_as_lm(self.ps2s_model_path)

        # If LM fine-tuning is requested, unfreeze PS2S parameters
        if self.hparams.lm_finetune_weight > 0.0 and self.ps2s_model is not None:
            for p in self.ps2s_model.parameters():
                p.requires_grad = True
            print("[ctc+lm] PS2S LM will be fine-tuned (lm_finetune_weight > 0).")
        else:
            if self.ps2s_model is not None:
                for p in self.ps2s_model.parameters():
                    p.requires_grad = False

        # phoneme label names for CTC ids (1..n_classes)
        default_labels = self._load_default_phoneme_labels()
        self.phoneme_labels = phoneme_labels or default_labels

        # metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_cer = MeanMetric()
        self.test_cer = MeanMetric()
        self._val_example_logged = False

    # ============================================================
    #             LOADERS
    # ============================================================

    def _load_gru_weights(self, ckpt_path: str) -> None:
        path = Path(ckpt_path)
        if not path.exists():
            print(f"[ctc+lm] GRU checkpoint not found: {path}")
            return

        # safe unpickling globals (PyTorch 2.6+)
        safe_list = [functools.partial]
        for cand in ["optim.AdamW", "optim.adamw.AdamW", "optim._adamw.AdamW"]:
            try:
                mod = torch
                for p in cand.split("."):
                    mod = getattr(mod, p)
                safe_list.append(mod)
            except Exception:
                pass
        for cls in safe_list:
            try:
                torch.serialization.add_safe_globals([cls])
            except Exception:
                pass

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[ctc+lm] Error loading GRU checkpoint: {e}")
            return

        state = ckpt.get("state_dict", ckpt)
        filtered = {k.replace("net.", ""): v for k, v in state.items() if k.startswith("net.")}

        if not filtered:
            print("[ctc+lm] WARNING: no net.* parameters found in checkpoint.")
            return

        missing, unexpected = self.net.load_state_dict(filtered, strict=False)
        if missing:
            print(f"[ctc+lm] GRU missing keys: {missing}")
        if unexpected:
            print(f"[ctc+lm] GRU unexpected keys: {unexpected}")
        print(f"[ctc+lm] GRU weights loaded from {path}")

    def _load_default_phoneme_labels(self) -> Optional[List[str]]:
        """
        Reuse the phoneme label list defined in configs/model/LLMSpeech.yaml.
        """
        cfg_path = Path(__file__).resolve().parents[2] / "configs" / "model" / "LLMSpeech.yaml"
        if not cfg_path.exists():
            return None
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            labels = data.get("phoneme_labels") if isinstance(data, dict) else None
            if isinstance(labels, list):
                return labels
        except Exception:
            return None
        return None

    def _load_ps2s_as_lm(self, ckpt_path: str) -> None:
        path = Path(ckpt_path)
        if not path.exists():
            print(f"[ctc+lm] WARNING: PS2S checkpoint not found at {path}. LM will be disabled.")
            return

        ckpt = torch.load(path, map_location="cpu")

        phoneme_to_idx: Dict[str, int] = ckpt["phoneme_to_idx"]
        idx_to_phoneme = ckpt["idx_to_phoneme"]
        pad_idx = int(ckpt["pad_idx"])
        sos_idx = int(ckpt["sos_idx"])
        eos_idx = int(ckpt["eos_idx"])
        max_len = int(ckpt["max_len"])
        model_kwargs = ckpt["model_kwargs"]

        vocab_size = model_kwargs["vocab_size"]
        d_model = model_kwargs["d_model"]
        nhead = model_kwargs["nhead"]
        num_encoder_layers = model_kwargs["num_encoder_layers"]
        num_decoder_layers = model_kwargs["num_decoder_layers"]
        dim_feedforward = model_kwargs["dim_feedforward"]
        dropout = model_kwargs["dropout"]

        ps2s = Seq2SeqTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bidirectional=True,
            pad_idx=pad_idx,
        )
        ps2s.load_state_dict(ckpt["state_dict"], strict=True)
        ps2s.eval()

        self.ps2s_model = ps2s
        self.ps2s_meta = {
            "phoneme_to_idx": phoneme_to_idx,
            "idx_to_phoneme": idx_to_phoneme,
            "pad_idx": pad_idx,
            "sos_idx": sos_idx,
            "eos_idx": eos_idx,
            "max_len": max_len,
        }
        print(f"[ctc+lm] Loaded PS2S LM from {path}")

    def _ensure_ps2s_on_device(self, device: torch.device) -> None:
        if self.ps2s_model is not None:
            self.ps2s_model.to(device)

    # ============================================================
    #             HELPER FUNCTIONS
    # ============================================================

    def _compute_input_lengths(self, neural_lens: torch.Tensor) -> torch.Tensor:
        stride = self.net.strideLen
        kernel = self.net.kernelLen
        lengths = torch.clamp(neural_lens - kernel, min=0)
        lengths = torch.div(lengths, stride, rounding_mode="floor")
        return lengths.to(torch.int32)

    def _ctc_greedy_decode(
        self,
        logits: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> List[List[int]]:
        """
        Standard greedy CTC decoding on acoustic logits.
        logits: (B, T, C)
        returns: list of token-id sequences (no blanks, collapsed)
        """
        preds = logits.argmax(dim=-1)  # (B, T)
        seqs: List[List[int]] = []
        for seq, L in zip(preds, input_lengths):
            seq = seq[: int(L)]
            seq = torch.unique_consecutive(seq)
            seq = [int(t) for t in seq if t != 0]  # drop blank
            seqs.append(seq)
        return seqs

    def _collapse_tokens(self, tokens: List[int]) -> List[int]:
        """
        Collapse consecutive duplicates (CTC-style) but keep tokens otherwise unchanged.
        """
        collapsed: List[int] = []
        for t in tokens:
            if not collapsed or t != collapsed[-1]:
                collapsed.append(t)
        return collapsed

    def _collapse_batch(self, batch_tokens: List[List[int]]) -> List[List[int]]:
        return [self._collapse_tokens(seq) for seq in batch_tokens]

    def _token_error_rate(
        self,
        pred_tokens: List[List[int]],
        labels: torch.Tensor,
        label_lens: torch.Tensor,
    ) -> float:
        total_ed = 0
        total_len = 0
        for p, tgt, L in zip(pred_tokens, labels, label_lens):
            tgt_seq = tgt[: int(L)].tolist()
            matcher = SequenceMatcher(a=tgt_seq, b=p)
            total_ed += matcher.distance()
            total_len += len(tgt_seq)
        return 0.0 if total_len == 0 else total_ed / total_len

    # ============= MAPPING BETWEEN CTC IDS & PHONEMES ============= #

    def _ctc_ids_to_phonemes(self, token_ids: List[int]) -> List[str]:
        """
        CTC label IDs (1..N) → phoneme strings from self.phoneme_labels.
        """
        if not self.phoneme_labels:
            return [str(t) for t in token_ids]
        out: List[str] = []
        for t in token_ids:
            if 1 <= t <= len(self.phoneme_labels):
                out.append(self.phoneme_labels[t - 1])
            else:
                out.append("<unk>")
        return out

    def _phonemes_to_ctc_ids(self, phonemes: List[str]) -> List[int]:
        """
        Phoneme strings → CTC label IDs (1..N). Unknowns are skipped.
        """
        if not self.phoneme_labels:
            # fallback: assume phonemes are stringified ints
            out: List[int] = []
            for ph in phonemes:
                try:
                    out.append(int(ph))
                except ValueError:
                    continue
            return out

        mapping = {ph: i + 1 for i, ph in enumerate(self.phoneme_labels)}
        out: List[int] = []
        for ph in phonemes:
            if ph in mapping:
                out.append(mapping[ph])
        return out

    # ============= PS2S TOKENIZATION HELPERS ============= #

    def _phonemes_to_ps2s_src(self, phonemes: List[str]) -> List[int]:
        """
        Map phoneme strings (e.g. ["DH", "AH", ...]) to PS2S token ids for src.
        """
        assert self.ps2s_meta is not None
        phoneme_to_idx: Dict[str, int] = self.ps2s_meta["phoneme_to_idx"]
        pad_idx = self.ps2s_meta["pad_idx"]
        max_len = min(self.ps2s_max_len, self.ps2s_meta["max_len"])

        toks = [phoneme_to_idx.get(ph, pad_idx) for ph in phonemes]
        toks = toks[: max_len]
        if not toks:
            toks = [pad_idx]
        return toks

    # ============================================================
    #             PS2S GREEDY CORRECTION (POST-CORRECTOR)
    # ============================================================

    def _ps2s_greedy_correct(
        self,
        noisy_phonemes: List[str],
        device: torch.device,
    ) -> List[str]:
        """
        Use PS2S as a denoising LM:

          noisy phonemes (from CTC) → PS2S → corrected phoneme sequence

        - src = PS2S tokens from noisy_phonemes
        - decoder runs autoregressively with <sos> → <eos>
        """

        if self.ps2s_model is None or self.ps2s_meta is None:
            # no LM available
            return noisy_phonemes

        self._ensure_ps2s_on_device(device)
        self.ps2s_model.eval()

        meta = self.ps2s_meta
        phoneme_to_idx: Dict[str, int] = meta["phoneme_to_idx"]
        idx_to_phoneme = meta["idx_to_phoneme"]
        pad_idx = meta["pad_idx"]
        sos_idx = meta["sos_idx"]
        eos_idx = meta["eos_idx"]
        max_len = min(self.ps2s_max_len, meta["max_len"])

        # ----- src tokens from noisy phonemes -----
        src_tokens = self._phonemes_to_ps2s_src(noisy_phonemes)
        src = torch.tensor(src_tokens, device=device, dtype=torch.long).unsqueeze(0)  # (1, S)

        # ----- greedy autoregressive decoding -----
        T = max_len
        tgt_tokens = [sos_idx]  # start with <sos>

        with torch.no_grad():
            for _ in range(T - 1):
                tgt_in = torch.tensor(tgt_tokens, device=device, dtype=torch.long).unsqueeze(0)  # (1, t)
                logits = self.ps2s_model(src, tgt_in)  # (1, t, V)
                next_logits = logits[0, -1, :]
                next_tok = int(next_logits.argmax(dim=-1).item())
                if next_tok == eos_idx:
                    break
                if next_tok == pad_idx:
                    break
                tgt_tokens.append(next_tok)

        # map PS2S token ids back to phoneme strings
        corrected: List[str] = []
        for tok in tgt_tokens[1:]:  # skip <sos>
            if tok in (sos_idx, eos_idx, pad_idx):
                continue
            if isinstance(idx_to_phoneme, dict):
                ph = idx_to_phoneme.get(tok, "<unk>")
            else:
                if 0 <= tok < len(idx_to_phoneme):
                    ph = idx_to_phoneme[tok]
                else:
                    ph = "<unk>"
            corrected.append(ph)

        if not corrected:
            return noisy_phonemes
        return corrected

    def _decode_with_ps2s_post(
        self,
        logits: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Simple pipeline:
            logits → greedy CTC → noisy phonemes → PS2S greedy → corrected phonemes → CTC ids
        """
        noisy_ids_batch = self._ctc_greedy_decode(logits, input_lengths)

        if self.ps2s_model is None or self.ps2s_meta is None:
            return noisy_ids_batch, noisy_ids_batch

        corrected_ids_batch: List[List[int]] = []
        device = logits.device

        for noisy_ids in noisy_ids_batch:
            noisy_ph = self._ctc_ids_to_phonemes(noisy_ids)
            corrected_ph = self._ps2s_greedy_correct(noisy_ph, device=device)
            corrected_ids = self._phonemes_to_ctc_ids(corrected_ph)
            corrected_ids_batch.append(corrected_ids)

        return noisy_ids_batch, corrected_ids_batch

    # ============================================================
    #       PS2S SEQUENCE LOGPROB (FOR SHALLOW FUSION)
    # ============================================================

    def _ps2s_sequence_logprob(
        self,
        candidate_phonemes: List[str],
        anchor_phonemes: List[str],
        device: torch.device,
    ) -> float:
        """
        Compute log P_LM(candidate | anchor) under PS2S using teacher forcing.

        - src = noisy anchor phonemes (e.g., greedy CTC output)
        - tgt = candidate sequence we want to score
        """
        if self.ps2s_model is None or self.ps2s_meta is None:
            return 0.0

        self._ensure_ps2s_on_device(device)
        meta = self.ps2s_meta
        phoneme_to_idx: Dict[str, int] = meta["phoneme_to_idx"]
        pad_idx = meta["pad_idx"]
        sos_idx = meta["sos_idx"]
        eos_idx = meta["eos_idx"]
        max_len = min(self.ps2s_max_len, meta["max_len"])

        # ---- src from anchor ----
        src_tokens = self._phonemes_to_ps2s_src(anchor_phonemes)
        src_tensor = torch.tensor(src_tokens, device=device, dtype=torch.long).unsqueeze(0)  # (1, S)

        # ---- target tokens from candidate ----
        toks = [phoneme_to_idx.get(ph, pad_idx) for ph in candidate_phonemes]
        if not toks:
            return float("-inf")

        toks = toks[: max_len - 1]  # leave room for <eos>
        tgt_in = [sos_idx] + toks
        tgt_out = toks + [eos_idx]
        T = len(tgt_out)

        tgt_in_tensor = torch.tensor(tgt_in, device=device, dtype=torch.long).unsqueeze(0)
        logits = self.ps2s_model(src_tensor, tgt_in_tensor)  # (1, T_in, V)
        log_probs = logits.log_softmax(dim=-1)

        T_model = log_probs.size(1)
        T_eff = min(T, T_model)
        tgt_out_tensor = torch.tensor(tgt_out[:T_eff], device=device, dtype=torch.long).unsqueeze(0)

        # gather log probs at target tokens
        # shape: (1, T_eff)
        log_p = torch.gather(log_probs[:, :T_eff, :], 2, tgt_out_tensor.unsqueeze(-1)).squeeze(-1)
        return float(log_p.sum().item())

    # ============================================================
    #       CTC BEAM SEARCH + SHALLOW FUSION RESCORING
    # ============================================================

    def _ctc_beam_search_one_utt(
        self,
        log_probs: torch.Tensor,  # (T, C) log-softmaxed
        input_len: int,
        beam_size: int,
        blank_idx: int = 0,
    ) -> List[Tuple[List[int], float]]:
        """
        Simple CTC beam search that keeps a beam of candidate token prefixes.
        Approximate but practical; sequences are built by appending non-blank tokens.
        """
        T, C = log_probs.shape
        T = int(input_len)

        # beam: dict[prefix tuple] = log_score
        beam: Dict[Tuple[int, ...], float] = {(): 0.0}

        for t in range(T):
            frame = log_probs[t]  # (C,)
            # restrict to top-K symbols for efficiency
            top_k = min(beam_size, C)
            vals, idxs = frame.topk(top_k)
            vals = vals.tolist()
            idxs = idxs.tolist()

            new_beam: Dict[Tuple[int, ...], float] = {}

            for prefix, score in beam.items():
                for c, log_p in zip(idxs, vals):
                    c = int(c)
                    if c == blank_idx:
                        # staying in same prefix
                        new_score = score + log_p
                        prev = new_beam.get(prefix, float("-inf"))
                        new_beam[prefix] = float(torch.logaddexp(
                            torch.tensor(prev), torch.tensor(new_score)
                        ).item())
                    else:
                        new_prefix = prefix + (c,)
                        new_score = score + log_p
                        prev = new_beam.get(new_prefix, float("-inf"))
                        new_beam[new_prefix] = float(torch.logaddexp(
                            torch.tensor(prev), torch.tensor(new_score)
                        ).item())

            # prune beam
            beam = dict(sorted(new_beam.items(), key=lambda kv: kv[1], reverse=True)[:beam_size])

        # convert prefixes to sequences (they already exclude blanks here)
        results: List[Tuple[List[int], float]] = []
        for prefix, score in beam.items():
            seq = list(prefix)
            results.append((seq, score))
        return results

    def _decode_shallow_fusion(
        self,
        logits: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Shallow-fusion decoding:

          - For each utterance:
              * run CTC beam search -> N best sequences with acoustic scores
              * get a greedy CTC decode as noisy anchor
              * compute PS2S log P_LM(candidate | anchor)
              * fused_score = log P_CTC + lm_weight * log P_LM
              * choose best fused candidate
        """
        device = logits.device
        log_probs = logits.log_softmax(dim=-1)  # (B, T, C)
        batch_size = logits.size(0)

        # greedy decode for noisy anchors
        greedy_ids_batch = self._ctc_greedy_decode(logits, input_lengths)

        final_ids_batch: List[List[int]] = []

        for i in range(batch_size):
            T_i = int(input_lengths[i].item())
            lp_i = log_probs[i, :T_i, :]  # (T_i, C)

            # pure CTC beam
            candidates = self._ctc_beam_search_one_utt(
                lp_i,
                input_len=T_i,
                beam_size=self.hparams.beam_size,
                blank_idx=0,
            )

            anchor_ph = self._ctc_ids_to_phonemes(greedy_ids_batch[i])
            best_score = float("-inf")
            best_seq = greedy_ids_batch[i]  # fallback

            for seq_ids, ctc_score in candidates:
                cand_ph = self._ctc_ids_to_phonemes(seq_ids)
                if self.ps2s_model is not None and self.ps2s_meta is not None and self.hparams.lm_weight != 0.0:
                    lm_logprob = self._ps2s_sequence_logprob(
                        candidate_phonemes=cand_ph,
                        anchor_phonemes=anchor_ph,
                        device=device,
                    )
                else:
                    lm_logprob = 0.0

                fused_score = ctc_score + self.hparams.lm_weight * lm_logprob
                if fused_score > best_score:
                    best_score = fused_score
                    best_seq = seq_ids

            final_ids_batch.append(best_seq)

        return greedy_ids_batch, final_ids_batch

    # ============================================================
    #        LM FINE-TUNING LOSS (TEXT-ONLY, ON LABELS)
    # ============================================================

    def _compute_lm_finetune_loss(
        self,
        labels: torch.Tensor,
        label_lens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Optional text-only LM fine-tuning on PS2S using BMI phoneme labels.

        For each sample:
          - Convert label IDs -> phoneme strings.
          - Map to PS2S vocab tokens.
          - Use identity mapping (src = clean phonemes) for adaptation.
          - Teacher-forced NLL over target sequence.
        """
        if (
            self.ps2s_model is None
            or self.ps2s_meta is None
            or self.hparams.lm_finetune_weight <= 0.0
        ):
            return torch.zeros((), device=device)

        self._ensure_ps2s_on_device(device)
        meta = self.ps2s_meta
        phoneme_to_idx: Dict[str, int] = meta["phoneme_to_idx"]
        pad_idx = meta["pad_idx"]
        sos_idx = meta["sos_idx"]
        eos_idx = meta["eos_idx"]
        max_len = min(self.ps2s_max_len, meta["max_len"])

        total_nll = torch.zeros((), device=device)
        total_tokens = 0

        for seq, L in zip(labels, label_lens):
            ids = seq[: int(L)].tolist()
            phs = self._ctc_ids_to_phonemes(ids)
            toks = [phoneme_to_idx.get(ph, pad_idx) for ph in phs]
            if not toks:
                continue

            toks = toks[: max_len - 1]  # leave room for eos
            tgt_in = [sos_idx] + toks
            tgt_out = toks + [eos_idx]
            T = len(tgt_out)

            src_tokens = toks[: max_len]  # identity source for adaptation

            src_tensor = torch.tensor(src_tokens, device=device, dtype=torch.long).unsqueeze(0)
            tgt_in_tensor = torch.tensor(tgt_in, device=device, dtype=torch.long).unsqueeze(0)

            logits = self.ps2s_model(src_tensor, tgt_in_tensor)
            log_probs = logits.log_softmax(dim=-1)

            T_model = log_probs.size(1)
            T_eff = min(T, T_model)
            if T_eff <= 0:
                continue

            tgt_out_tensor = torch.tensor(tgt_out[:T_eff], device=device, dtype=torch.long).unsqueeze(0)
            # NLL for this sample
            log_p = torch.gather(
                log_probs[:, :T_eff, :],
                2,
                tgt_out_tensor.unsqueeze(-1),
            ).squeeze(-1)  # (1, T_eff)
            nll = -log_p.sum()

            total_nll = total_nll + nll
            total_tokens += T_eff

        if total_tokens == 0:
            return torch.zeros((), device=device)

        return total_nll / total_tokens

    # ============================================================
    #             FORWARD & LOSS
    # ============================================================

    def forward(self, neural: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        """
        Forward through the GRU acoustic model.
        """
        return self.net(neural, days)

    def _compute_loss(
        self,
        neural: torch.Tensor,
        labels: torch.Tensor,
        neural_lens: torch.Tensor,
        label_lens: torch.Tensor,
        days: torch.Tensor,
    ):
        """
        Training loss:

          L_total = L_CTC + lm_finetune_weight * L_LM   (if enabled)
        """
        logits = self.forward(neural, days)  # (B, T, C)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, C) for CTC

        input_lengths = self._compute_input_lengths(neural_lens).to(logits.device)
        label_lens = label_lens.to(logits.device)

        ctc = self.ctc_loss(log_probs, labels, input_lengths, label_lens)

        lm_loss = torch.zeros((), device=logits.device)
        if self.training and self.hparams.lm_finetune_weight > 0.0:
            lm_loss = self._compute_lm_finetune_loss(labels, label_lens, logits.device)

        total_loss = ctc + self.hparams.lm_finetune_weight * lm_loss
        return total_loss, logits, input_lengths, ctc, lm_loss

    # ============================================================
    #             LIGHTNING HOOKS
    # ============================================================

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_cer.reset()
        self.test_cer.reset()
        self._val_example_logged = False

    def on_validation_epoch_start(self) -> None:
        self._val_example_logged = False

    def training_step(self, batch, batch_idx):
        neural, labels, neural_lens, label_lens, days = batch
        total_loss, _, _, ctc_loss, lm_loss = self._compute_loss(
            neural, labels, neural_lens, label_lens, days
        )
        self.train_loss(total_loss)
        self.log("train/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ctc_loss", ctc_loss, on_step=False, on_epoch=True)
        if self.hparams.lm_finetune_weight > 0.0 and self.ps2s_model is not None:
            self.log("train/lm_loss", lm_loss, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        neural, labels, neural_lens, label_lens, days = batch
        total_loss, logits, input_lengths, _, _ = self._compute_loss(
            neural, labels, neural_lens, label_lens, days
        )
        self.val_loss(total_loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            # choose decoding mode
            if (
                self.ps2s_model is not None
                and self.ps2s_meta is not None
                and self.hparams.use_shallow_fusion
                and self.hparams.lm_weight != 0.0
            ):
                noisy_ids_batch, final_ids_batch = self._decode_shallow_fusion(
                    logits, input_lengths
                )
            elif self.ps2s_model is not None and self.ps2s_meta is not None:
                noisy_ids_batch, final_ids_batch = self._decode_with_ps2s_post(
                    logits, input_lengths
                )
            else:
                noisy_ids_batch = self._ctc_greedy_decode(logits, input_lengths)
                final_ids_batch = noisy_ids_batch

            # Ensure sequences are CTC-collapsed before scoring.
            final_ids_batch = self._collapse_batch(final_ids_batch)

            cer = self._token_error_rate(final_ids_batch, labels, label_lens)
            self.val_cer(cer)
            self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

            # Example logging for inspection
            if not self._val_example_logged and final_ids_batch:
                tgt_len = int(label_lens[0].item())
                tgt_seq = labels[0, :tgt_len].tolist()

                noisy_ph = self._ctc_ids_to_phonemes(noisy_ids_batch[0])
                final_ph = self._ctc_ids_to_phonemes(final_ids_batch[0])
                tgt_ph = [
                    self.phoneme_labels[t - 1]
                    if self.phoneme_labels and 1 <= t <= len(self.phoneme_labels)
                    else str(t)
                    for t in tgt_seq
                ]

                self.print(
                    "[val example] noisy:     " + " ".join(noisy_ph)
                    + "\n[val example] decoded:  " + " ".join(final_ph)
                    + "\n[val example] target:   " + " ".join(tgt_ph)
                )
                self._val_example_logged = True

    def test_step(self, batch, batch_idx):
        neural, labels, neural_lens, label_lens, days = batch
        total_loss, logits, input_lengths, _, _ = self._compute_loss(
            neural, labels, neural_lens, label_lens, days
        )
        self.test_loss(total_loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            if (
                self.ps2s_model is not None
                and self.ps2s_meta is not None
                and self.hparams.use_shallow_fusion
                and self.hparams.lm_weight != 0.0
            ):
                _, final_ids_batch = self._decode_shallow_fusion(logits, input_lengths)
            elif self.ps2s_model is not None and self.ps2s_meta is not None:
                _, final_ids_batch = self._decode_with_ps2s_post(logits, input_lengths)
            else:
                final_ids_batch = self._ctc_greedy_decode(logits, input_lengths)

            final_ids_batch = self._collapse_batch(final_ids_batch)

            cer = self._token_error_rate(final_ids_batch, labels, label_lens)
            self.test_cer(cer)
            self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        if self.hparams.optimizer is None:
            return None

        # all trainable parameters: GRU + (optionally) PS2S
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def train(self, mode: bool = True):
        super().train(mode)
        # keep PS2S in train/eval depending on whether we're fine-tuning it
        if self.ps2s_model is not None:
            if self.hparams.lm_finetune_weight > 0.0:
                self.ps2s_model.train(mode)
            else:
                self.ps2s_model.eval()
        return self

    def eval(self):
        super().eval()
        if self.ps2s_model is not None:
            self.ps2s_model.eval()
        return self


# ============================================================
#             SANITY CHECK
# ============================================================

if __name__ == "__main__":
    class DummyGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.strideLen = 4
            self.kernelLen = 16
            self.gru = nn.GRU(256, 128, batch_first=True)
            self.fc = nn.Linear(128, 41)  # 0 = blank, 1..40 = phonemes

        def forward(self, x, days):
            h, _ = self.gru(x)
            return self.fc(h)

    dummy = DummyGRU()
    _ = SpeechModuleCTCWithPS2SLM(
        net=dummy,
        optimizer=None,
        scheduler=None,
        compile=False,
        use_shallow_fusion=True,
        beam_size=8,
        lm_weight=0.3,
        lm_finetune_weight=0.0,
    )
    print("Sanity check OK (CTC + PS2S LM with shallow fusion & optional LM fine-tuning).")
