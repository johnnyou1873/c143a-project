# ============================================
#   GRU → frozen inference-only
#   Seq2Seq Transformer → trainable denoising LM
#   Training: NO teacher forcing (autoregressive)
#   Validation: decode ALL examples and compute CER
# ============================================

from typing import Any, Dict, List, Optional, Tuple
import math
import functools

import torch
from torch import nn
from lightning import LightningModule
from torchmetrics import MeanMetric
from edit_distance import SequenceMatcher

from .components.gru_diphone import PHONEME_LABELS

torch.set_float32_matmul_precision("medium")


# ------------------ utility masks ------------------ #
def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Standard causal mask (no attention to future positions)."""
    mask = torch.full((sz, sz), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    mask.fill_diagonal_(0.0)
    return mask


def _generate_limited_future_mask(
    sz: int, device: torch.device, lookahead: int = 1
) -> torch.Tensor:
    """
    Allow attention to current position and up to `lookahead` future steps;
    disallow further lookahead.
    """
    i = torch.arange(sz, device=device).unsqueeze(1)
    j = torch.arange(sz, device=device)
    return torch.where(j - i > lookahead, float("-inf"), 0.0)


# ------------------ positional encoding ------------------ #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1)]


# ------------------ Seq2Seq Transformer ------------------ #
class Seq2SeqTransformer(nn.Module):
    """
    Transformer encoder-decoder used as a phoneme-to-phoneme correction LM.

    - src: noisy phoneme tokens (already in LM vocab)
    - tgt: (optionally) clean phoneme tokens (LM vocab)
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
        pad_idx: int,
        attention_mode: str = "limited",  # 'bidir', 'limited', or 'causal'
        lookahead: int = 1,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.attention_mode = attention_mode
        self.lookahead = lookahead

        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    # ----- masks -----
    def _make_src_mask(self, src: torch.Tensor) -> Optional[torch.Tensor]:
        T = src.size(1)
        if self.attention_mode == "bidir":
            return None
        if self.attention_mode == "limited":
            return _generate_limited_future_mask(T, src.device, self.lookahead)
        return _generate_square_subsequent_mask(T, src.device)

    def _make_tgt_mask(self, tgt: torch.Tensor) -> Optional[torch.Tensor]:
        T = tgt.size(1)
        if self.attention_mode == "bidir":
            return None
        if self.attention_mode == "limited":
            return _generate_limited_future_mask(T, tgt.device, self.lookahead)
        return _generate_square_subsequent_mask(T, tgt.device)

    # ----- convenience encode / decode -----
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        src_pad_mask = src == self.pad_idx
        src_h = self.encoder(
            self.pos_enc(self.src_emb(src)),
            mask=self._make_src_mask(src),
            src_key_padding_mask=src_pad_mask,
        )
        return src_h

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard teacher-forced forward (not used for training in this LM fine-tune
        path, but kept for compatibility / debugging).
        """
        src_pad_mask = src == self.pad_idx
        tgt_pad_mask = tgt_in == self.pad_idx
        tgt_mask = self._make_tgt_mask(tgt_in)

        src_h = self.encoder(
            self.pos_enc(self.src_emb(src)),
            mask=self._make_src_mask(src),
            src_key_padding_mask=src_pad_mask,
        )
        tgt_h = self.decoder(
            self.pos_enc(self.tgt_emb(tgt_in)),
            src_h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.output_proj(tgt_h)


# ============================================================
#     MAIN LIGHTNING MODULE: GRU frozen → seq2seq trainable
# ============================================================
class SpeechModuleCTCWithPS2SLM(LightningModule):
    """
    LightningModule that:

    1. Uses a frozen GRU CTC model to map neural features → noisy phoneme IDs.
    2. Uses a trainable seq2seq Transformer LM to denoise / correct phoneme sequences.
    3. Trains the LM WITHOUT teacher forcing (autoregressive decoding).
    4. Validation decodes every example in the batch and logs CER.
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer,
        scheduler=None,
        gru_ckpt_path: str = "",
        ps2s_model_path: str = "",
        ps2s_max_len: int = 128,
        attention_mode: str = "bidir",
        lookahead: int = 1,
        phoneme_labels: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Save hydra-configurable hyperparameters, excluding the instantiated GRU net.
        self.save_hyperparameters(ignore=["net"])

        # -----------------------------
        # 1. Load and freeze GRU backbone
        # -----------------------------
        self.net = net
        if gru_ckpt_path:
            self._load_gru_weights(gru_ckpt_path)
        self._freeze_gru()

        # -----------------------------
        # 2. Load trainable seq2seq LM
        # -----------------------------
        (
            self.seq2seq,
            self.phoneme_to_idx,
            self.idx_to_phoneme,
            self.pad_idx,
            self.sos_idx,
            self.eos_idx,
            ckpt_max_len,
        ) = self._load_ps2s(ps2s_model_path, attention_mode, lookahead)

        # fine-tuning length cap
        self.max_len = min(ps2s_max_len, ckpt_max_len)

        # Train ONLY the seq2seq LM
        for p in self.seq2seq.parameters():
            p.requires_grad = True

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        # phoneme labels from CTC space (GRU outputs)
        self.phoneme_labels = phoneme_labels or PHONEME_LABELS

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_cer = MeanMetric()

    # ============================================================
    #  GRU: load & freeze
    # ============================================================
    def _load_gru_weights(self, path: str) -> None:
        try:
            with torch.serialization.safe_globals([functools.partial]):
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            torch.serialization.add_safe_globals([functools.partial])
            ckpt = torch.load(path, map_location="cpu", weights_only=False)

        state_dict = ckpt["state_dict"]
        clean = {k.replace("net.", ""): v for k, v in state_dict.items() if k.startswith("net.")}
        self.net.load_state_dict(clean, strict=False)

    def _freeze_gru(self) -> None:
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()

    # ============================================================
    #  PS2S loading
    # ============================================================
    def _load_ps2s(
        self,
        path: str,
        attention_mode: str,
        lookahead: int,
    ) -> Tuple[Seq2SeqTransformer, Dict[str, int], Dict[int, str], int, int, int, int]:
        try:
            with torch.serialization.safe_globals([functools.partial]):
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            torch.serialization.add_safe_globals([functools.partial])
            ckpt = torch.load(path, map_location="cpu", weights_only=False)

        model_kwargs = ckpt["model_kwargs"]
        pad_idx = ckpt["pad_idx"]

        model = Seq2SeqTransformer(
            vocab_size=model_kwargs["vocab_size"],
            d_model=model_kwargs["d_model"],
            nhead=model_kwargs["nhead"],
            num_encoder_layers=model_kwargs["num_encoder_layers"],
            num_decoder_layers=model_kwargs["num_decoder_layers"],
            dim_feedforward=model_kwargs["dim_feedforward"],
            dropout=model_kwargs["dropout"],
            pad_idx=pad_idx,
            attention_mode=attention_mode,
            lookahead=lookahead,
        )
        sd = ckpt["state_dict"]
        remapped = {}
        for k, v in sd.items():
            if k.startswith("proj."):
                remapped[k.replace("proj.", "output_proj.")] = v
            else:
                remapped[k] = v
        model.load_state_dict(remapped, strict=False)

        phoneme_to_idx: Dict[str, int] = ckpt["phoneme_to_idx"]
        # Ensure idx_to_phoneme keys are int
        idx_to_phoneme: Dict[int, str] = {
            int(k): v for k, v in ckpt["idx_to_phoneme"].items()
        }

        sos_idx = ckpt["sos_idx"]
        eos_idx = ckpt["eos_idx"]
        max_len = ckpt.get("max_len", 128)

        return model, phoneme_to_idx, idx_to_phoneme, pad_idx, sos_idx, eos_idx, max_len

    # ============================================================
    #  Core length + decoding utilities
    # ============================================================
    def _compute_output_lengths(self, neural_lens: torch.Tensor) -> torch.Tensor:
        """
        Map input neural_lens (pre-conv/GRU) to output CTC time-steps.

        Assumes self.net has attributes:
          - strideLen
          - kernelLen
        """
        stride = getattr(self.net, "strideLen", 1)
        kernel = getattr(self.net, "kernelLen", 1)
        effective = torch.clamp(neural_lens - kernel, min=0)
        return torch.div(effective, stride, rounding_mode="floor")

    def _ctc_decode(self, logits: torch.Tensor, lengths: torch.Tensor) -> List[List[int]]:
        """
        Simple greedy CTC decode with collapsing repeats and dropping blank (0).
        """
        preds = logits.argmax(-1)  # [B, T]
        seqs: List[List[int]] = []
        for seq, L in zip(preds, lengths):
            seq = seq[: int(L)]
            seq = torch.unique_consecutive(seq)
            seqs.append([int(x) for x in seq.tolist() if int(x) != 0])
        return seqs

    def _phoneme_labels_lookup(self, idx: int) -> Optional[str]:
        """
        Map CTC label ID → phoneme string using PHONEME_LABELS.
        Assumes:
           - 0 is blank, so phonemes start at 1.
           - PHONEME_LABELS[0] corresponds to label 1, etc.
        """
        zero = idx - 1
        if 0 <= zero < len(self.phoneme_labels):
            return self.phoneme_labels[zero]
        return None

    def _ids_to_seq2seq_tokens(self, ctc_ids: List[int]) -> List[int]:
        """
        Convert a sequence of CTC label IDs → seq2seq LM token IDs.
        Drops any labels that cannot be mapped into the LM vocab.
        """
        out: List[int] = []
        for t in ctc_ids:
            if t <= 0:
                continue
            ph = self._phoneme_labels_lookup(t)
            if ph is None:
                continue
            idx = self.phoneme_to_idx.get(ph)
            if idx is not None:
                out.append(idx)
        return out

    def _build_src_batch(self, noisy_ids: List[List[int]], device: torch.device) -> torch.Tensor:
        """
        Build src batch for LM from list of noisy CTC ID sequences.
        Each sequence is:
            [tokens mapped to LM vocab] + [eos]
        """
        src_list: List[List[int]] = []
        max_len = 0
        for seq in noisy_ids:
            toks = self._ids_to_seq2seq_tokens(seq)
            toks = toks[: self.max_len - 1]  # leave room for EOS
            toks.append(self.eos_idx)
            src_list.append(toks)
            max_len = max(max_len, len(toks))

        src = torch.full(
            (len(src_list), max_len),
            self.pad_idx,
            dtype=torch.long,
            device=device,
        )
        for i, s in enumerate(src_list):
            src[i, : len(s)] = torch.tensor(s, device=device)
        return src

    # ============================================================
    #  Autoregressive LM forward (NO teacher forcing)
    # ============================================================
    def _autoregressive_forward(
        self,
        src: torch.Tensor,
        return_tokens: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Autoregressive decoding:

        - Encode src once.
        - Start from <sos>.
        - At each step, feed previous predictions back in.
        - Stop when all sequences emitted <eos> or reach self.max_len.

        Returns:
            logits: [B, T_pred, V]
            token_seqs (optional): [B, T_pred] int tensor (without <sos>)
        """
        device = src.device
        src_pad_mask = src == self.pad_idx

        memory = self.seq2seq.encoder(
            self.seq2seq.pos_enc(self.seq2seq.src_emb(src)),
            mask=self.seq2seq._make_src_mask(src),
            src_key_padding_mask=src_pad_mask,
        )

        B = src.size(0)
        ys = torch.full(
            (B, 1), self.sos_idx, dtype=torch.long, device=device
        )  # [B, 1] with <sos>
        logits_steps: List[torch.Tensor] = []

        for _ in range(self.max_len):
            tgt_mask = self.seq2seq._make_tgt_mask(ys)
            tgt_pad_mask = ys == self.pad_idx

            dec_h = self.seq2seq.decoder(
                self.seq2seq.pos_enc(self.seq2seq.tgt_emb(ys)),
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            step_logits = self.seq2seq.output_proj(dec_h[:, -1:])  # [B, 1, V]
            logits_steps.append(step_logits)

            next_token = step_logits.argmax(-1)  # [B, 1]

            ys = torch.cat([ys, next_token], dim=1)  # append predicted token
            # If all sequences predicted EOS, we can stop early
            if (next_token == self.eos_idx).all():
                break

        logits = torch.cat(logits_steps, dim=1)  # [B, T_pred, V]

        if not return_tokens:
            return logits, None

        # Remove the leading <sos>, keep everything up to T_pred tokens
        token_seqs = ys[:, 1 : 1 + logits.size(1)]  # [B, T_pred]
        return logits, token_seqs

    # ============================================================
    #  Compute LM loss (no teacher forcing)
    # ============================================================
    def _compute_loss(
        self,
        neural: torch.Tensor,
        labels: torch.Tensor,
        neural_lens: torch.Tensor,
        label_lens: torch.Tensor,
        days: torch.Tensor,
        return_preds: bool = False,
    ) -> Tuple[torch.Tensor, List[List[int]], List[List[int]], torch.Tensor, Optional[torch.Tensor]]:
        """
        End-to-end forward for a batch:

        1. Frozen GRU produces CTC logits.
        2. We greedy-decode logits into CTC label IDs (noisy phonemes).
        3. We map both noisy (from GRU) and clean (labels) into LM vocab.
        4. We run NO-TEACHER-FORCING autoregressive LM forward.
        5. We compute CE loss against clean LM tokens.
        """
        device = neural.device

        # ---- 1) GRU forward (no grad) ----
        with torch.no_grad():
            ctc_logits = self.net(neural, days)  # [B, T_ctc, C_ctc]
            out_lengths = self._compute_output_lengths(neural_lens)
            noisy_ctc_ids = self._ctc_decode(ctc_logits, out_lengths)

        # ---- 2) Convert clean labels into LM token IDs ----
        target_lm_ids: List[List[int]] = []
        for seq, L in zip(labels, label_lens):
            # labels are CTC IDs; convert to list and truncate to length L
            seq_ids = [int(t) for t in seq[: int(L)] if int(t) > 0]
            target_lm_ids.append(self._ids_to_seq2seq_tokens(seq_ids))

        # ---- 3) Build LM src batch ----
        src_batch = self._build_src_batch(noisy_ctc_ids, device=device)  # [B, T_src]

        # ---- 4) Autoregressive LM forward ----
        self.seq2seq.train()
        pred_logits, pred_token_seqs = self._autoregressive_forward(
            src_batch, return_tokens=return_preds
        )  # [B, T_pred, V]

        # ---- 5) Build target tensor aligned to predicted length ----
        B, T_pred, V = pred_logits.shape
        tgt_batch = torch.full(
            (B, T_pred),
            self.pad_idx,
            dtype=torch.long,
            device=device,
        )
        for i, tgt_ids in enumerate(target_lm_ids):
            truncated = tgt_ids[:T_pred]
            if len(truncated) == 0:
                continue
            tgt_batch[i, : len(truncated)] = torch.tensor(truncated, device=device)

        loss = self.criterion(
            pred_logits.reshape(-1, V),
            tgt_batch.reshape(-1),
        )

        return loss, noisy_ctc_ids, target_lm_ids, src_batch, pred_token_seqs

    # ============================================================
    #  Training / Validation steps
    # ============================================================
    def training_step(self, batch: Tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        loss, _, _, _, _ = self._compute_loss(*batch, return_preds=False)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[Any, ...], batch_idx: int) -> None:
        # Compute loss and token-level predictions
        loss, noisy_ctc_ids, target_lm_ids, src_batch, pred_token_seqs = self._compute_loss(
            *batch, return_preds=True
        )
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, prog_bar=True, on_epoch=True)

        # If we have predicted token sequences, compute CER
        if pred_token_seqs is not None:
            preds_as_lists: List[List[int]] = []
            for row in pred_token_seqs:
                seq: List[int] = []
                for t in row.tolist():
                    if t in (self.pad_idx, self.sos_idx, self.eos_idx):
                        if t == self.eos_idx:
                            break
                        continue
                    seq.append(int(t))
                preds_as_lists.append(seq)

            cer = self._compute_cer(preds_as_lists, target_lm_ids)
            self.val_cer(cer)
            self.log("val/cer", self.val_cer, prog_bar=True, on_epoch=True)

            # Log first example in batch
            if len(src_batch) > 0 and len(target_lm_ids) > 0 and len(preds_as_lists) > 0:
                noisy_txt = " ".join(self._map_ids_to_symbols(src_batch[0]))
                tgt_txt = " ".join(
                    self.idx_to_phoneme.get(t, f"<{t}>") for t in target_lm_ids[0]
                )
                pred_txt = " ".join(
                    self.idx_to_phoneme.get(t, f"<{t}>") for t in preds_as_lists[0]
                )

                self.print(f"[val] noisy: {noisy_txt}")
                self.print(f"[val] pred : {pred_txt}")
                self.print(f"[val] tgt  : {tgt_txt}")

    # ============================================================
    #  CER + mapping utilities
    # ============================================================
    def _compute_cer(
        self,
        preds: List[List[int]],
        tgts: List[List[int]],
    ) -> float:
        """
        Character (here: token/phoneme) error rate:
            total edit distance / total target length
        """
        total_d = 0
        total_l = 0
        for p, t in zip(preds, tgts):
            if len(t) == 0:
                continue
            d = SequenceMatcher(a=t, b=p).distance()
            total_d += d
            total_l += len(t)
        return 0.0 if total_l == 0 else total_d / total_l

    def _map_ids_to_symbols(self, row: torch.Tensor) -> List[str]:
        """
        Map LM vocab IDs → phoneme strings, dropping pad/sos/eos.
        """
        out: List[str] = []
        for t in row.tolist():
            if t in (self.pad_idx, self.sos_idx, self.eos_idx):
                continue
            out.append(self.idx_to_phoneme.get(int(t), f"<{t}>"))
        return out

    # ============================================================
    #  Optimizer / scheduler
    # ============================================================
    def configure_optimizers(self):
        """
        Expect hydra config to pass in:

            optimizer: a callable taking parameters → torch.optim.Optimizer
            scheduler: (optional) a callable taking optimizer → lr_scheduler

        Only the seq2seq LM parameters are trainable / optimized.
        """
        opt = self.hparams.optimizer(self.seq2seq.parameters())
        if self.hparams.scheduler is None:
            return opt
        sch = self.hparams.scheduler(opt)
        return {
            "optimizer": opt,
            "lr_scheduler": sch,
        }
