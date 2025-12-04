from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import functools
import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from edit_distance import SequenceMatcher
import yaml


# ============================================================
#                  PHONEME-TO-PHONEME Seq2Seq
# ============================================================

class Seq2SeqCorrection(torch.nn.Module):
    """Small Transformer encoder-decoder used for phoneme-to-phoneme correction."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        pad_idx: int = 0,
        lookahead: int = 1,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.lookahead = lookahead

        self.src_emb = torch.nn.Embedding(vocab_size, d_model)
        self.tgt_emb = torch.nn.Embedding(vocab_size, d_model)

        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        dec_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )

        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.decoder = torch.nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.proj = torch.nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """src: (B, S) noisy phonemes,  tgt_in: (B, T) clean inputs shifted right"""

        src_key_padding_mask = src == self.pad_idx
        tgt_key_padding_mask = tgt_in == self.pad_idx

        # standard causal decoder mask
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            tgt_in.size(1), device=tgt_in.device
        )

        # LIMITED LOOKAHEAD MASK for encoder
        L = src.size(1)
        i = torch.arange(L, device=src.device).unsqueeze(1)
        j = torch.arange(L, device=src.device)
        src_mask = torch.where(
            j - i > self.lookahead,
            torch.tensor(float("-inf"), device=src.device),
            torch.tensor(0.0, device=src.device)
        )

        src_h = self.encoder(
            self.src_emb(src),
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        tgt_h = self.decoder(
            self.tgt_emb(tgt_in),
            src_h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.proj(tgt_h)


# ============================================================
#                  MAIN LIGHTNING MODULE
# ============================================================

class SpeechModuleWithPS2S(LightningModule):
    """LightningModule that combines a GRU CTC model with a noisy→clean phoneme corrector."""

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
        # ps2s
        gru_ckpt_path: str = "data/gru512.ckpt",
        ps2s_model_path: str = "data/phoneme_seq2seq.pt",
        ps2s_weight: float = 0.5,
        ps2s_max_len: int = 128,
        phoneme_labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        if net is None:
            raise ValueError("`net` must be provided.")

        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

        # ---------------- GRU LOADING -----------------
        self._load_gru_weights(gru_ckpt_path)

        # ---------------- CTC -----------------
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        # ---------------- PS2S -----------------
        self.ps2s_weight = ps2s_weight
        self.ps2s_max_len = ps2s_max_len
        self._ps2s_model_path = ps2s_model_path
        self.ps2s_model: Optional[Seq2SeqCorrection] = None
        self.ps2s_meta: Optional[Dict[str, Any]] = None
        self.ps2s_loss: Optional[torch.nn.Module] = None

        # load PS2S at init so its parameters are registered & optimized
        if self.ps2s_weight > 0:
            try:
                ps2s, meta = self._load_ps2s(self._ps2s_model_path)
            except FileNotFoundError:
                print(f"[ps2s] WARNING: PS2S checkpoint not found at {self._ps2s_model_path}. "
                      f"PS2S will be disabled.")
                self.ps2s_weight = 0.0
            else:
                self.ps2s_model = ps2s
                self.ps2s_meta = meta
                self.ps2s_loss = torch.nn.CrossEntropyLoss(ignore_index=meta["pad_idx"])

        # phoneme label names
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
            print(f"[ps2s] GRU checkpoint not found: {path}")
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
            print(f"[ps2s] Error loading GRU checkpoint: {e}")
            return

        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        filtered = {k.replace("net.", ""): v for k, v in state.items() if k.startswith("net.")}

        if not filtered:
            print("[ps2s] WARNING: no net.* parameters found.")
            return

        missing, unexpected = self.net.load_state_dict(filtered, strict=False)
        if missing:
            print(f"[ps2s] missing: {missing}")
        if unexpected:
            print(f"[ps2s] unexpected: {unexpected}")
        print(f"[ps2s] GRU weights loaded from {path}")

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

    # ============= PS2S LOADING ============= #

    def _load_ps2s(self, ckpt_path: str):
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(path)

        ckpt = torch.load(path, map_location="cpu")

        meta = {
            "phoneme_to_idx": ckpt["phoneme_to_idx"],
            "idx_to_phoneme": ckpt["idx_to_phoneme"],
            "pad_idx": int(ckpt["pad_idx"]),
            "sos_idx": int(ckpt["sos_idx"]),
            "eos_idx": int(ckpt["eos_idx"]),
            "max_len": int(ckpt["max_len"]),
        }

        ps2s = Seq2SeqCorrection(
            pad_idx=meta["pad_idx"],
            lookahead=1,
            **ckpt["model_kwargs"]
        )
        ps2s.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"[ps2s] PS2S weights loaded from {path}")
        return ps2s, meta

    def _ensure_ps2s_loaded(self, device):
        """Make sure PS2S is on the right device (parameters are already registered)."""
        if self.ps2s_weight > 0 and self.ps2s_model is not None:
            self.ps2s_model.to(device)

    # ============================================================
    #             HELPER FUNCTIONS
    # ============================================================

    def _compute_input_lengths(self, neural_lens):
        stride = self.net.strideLen
        kernel = self.net.kernelLen
        lengths = torch.clamp(neural_lens - kernel, min=0)
        lengths = torch.div(lengths, stride, rounding_mode="floor")
        return lengths.to(torch.int32)

    def _ctc_greedy_decode(self, logits, input_lengths):
        preds = logits.argmax(dim=-1)
        seqs = []
        for seq, L in zip(preds, input_lengths):
            seq = seq[: int(L)]
            seq = torch.unique_consecutive(seq)
            seq = [int(t) for t in seq if t != 0]
            seqs.append(seq)
        return seqs

    def _token_error_rate(self, pred, labels, lens):
        total_ed = 0
        total_len = 0
        for p, tgt, L in zip(pred, labels, lens):
            tgt = tgt[: int(L)].tolist()
            matcher = SequenceMatcher(a=tgt, b=p)
            total_ed += matcher.distance()
            total_len += len(tgt)
        return 0.0 if total_len == 0 else total_ed / total_len

    # ============= Convert preds to phonemes ============= #

    def _pred_tokens_to_phonemes(self, pred_tokens: List[int]) -> List[str]:
        if not self.phoneme_labels:
            return [str(t) for t in pred_tokens]
        out = []
        for t in pred_tokens:
            if 1 <= t <= len(self.phoneme_labels):
                out.append(self.phoneme_labels[t - 1])
            else:
                out.append("<unk>")
        return out

    # ============= Tokenization ============= #

    def _phonemes_to_ps2s_tokens(self, phonemes: List[str]):
        lookup = self.ps2s_meta["phoneme_to_idx"]
        sos = self.ps2s_meta["sos_idx"]
        eos = self.ps2s_meta["eos_idx"]
        pad = self.ps2s_meta["pad_idx"]
        max_len = min(self.ps2s_max_len, self.ps2s_meta["max_len"])

        toks = [lookup.get(ph, pad) for ph in phonemes]
        toks = toks[: max_len - 2]
        toks = [sos] + toks + [eos]
        toks += [pad] * (max_len - len(toks))

        tgt_in = toks[:-1]
        tgt_out = toks[1:]
        return toks, tgt_in, tgt_out

    # ============= Build PS2S batch ============= #

    def _prepare_ps2s_batch(self, noisy_preds, labels, lens, device):
        src_list = []
        tgt_in_list = []
        tgt_out_list = []

        for noisy, gt_seq, L in zip(noisy_preds, labels, lens):
            # 1. noisy phonemes from GRU/CTC
            noisy_phs = self._pred_tokens_to_phonemes(noisy)

            # 2. clean GT phonemes
            gt_trim = gt_seq[: int(L)].tolist()
            clean_phs = [
                self.phoneme_labels[t - 1] if 1 <= t <= len(self.phoneme_labels)
                else "<unk>"
                for t in gt_trim
            ]

            # tokenize both
            src_full, _, _ = self._phonemes_to_ps2s_tokens(noisy_phs)
            _, tgt_in, tgt_out = self._phonemes_to_ps2s_tokens(clean_phs)

            src_list.append(src_full)
            tgt_in_list.append(tgt_in)
            tgt_out_list.append(tgt_out)

        src = torch.tensor(src_list, device=device, dtype=torch.long)
        tgt_in = torch.tensor(tgt_in_list, device=device, dtype=torch.long)
        tgt_out = torch.tensor(tgt_out_list, device=device, dtype=torch.long)
        return src, tgt_in, tgt_out

    # ============================================================
    #             FORWARD & LOSS
    # ============================================================

    def forward(self, neural, days):
        return self.net(neural, days)

    def _compute_loss(self, neural, labels, neural_lens, label_lens, days):
        logits = self.forward(neural, days)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

        input_lengths = self._compute_input_lengths(neural_lens).to(logits.device)
        label_lens = label_lens.to(logits.device)

        # ------------- CTC -------------
        ctc = self.ctc_loss(log_probs, labels, input_lengths, label_lens)

        # ------------- PS2S -------------
        ps2s_loss = torch.zeros((), device=logits.device)

        if self.ps2s_weight > 0 and self.ps2s_model is not None:
            self._ensure_ps2s_loaded(logits.device)

            # noisy GRU predictions → transformer input
            noisy_pred = self._ctc_greedy_decode(logits, input_lengths)

            # build PS2S batch
            src, tgt_in, tgt_out = self._prepare_ps2s_batch(
                noisy_pred, labels, label_lens, logits.device
            )

            ps2s_logits = self.ps2s_model(src, tgt_in)

            ps2s_loss = self.ps2s_loss(
                ps2s_logits.reshape(-1, ps2s_logits.size(-1)),
                tgt_out.reshape(-1),
            )

        total_loss = ctc + self.ps2s_weight * ps2s_loss
        return total_loss, logits, input_lengths, ps2s_loss

    # ============================================================
    #             LIGHTNING HOOKS
    # ============================================================

    def on_train_start(self):
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_cer.reset()
        self.test_cer.reset()
        self._val_example_logged = False

    def on_validation_epoch_start(self):
        self._val_example_logged = False

    def training_step(self, batch, batch_idx):
        neural, labels, neural_lens, label_lens, days = batch
        loss, _, _, ps2s_loss = self._compute_loss(neural, labels, neural_lens, label_lens, days)
        self.train_loss(loss)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.ps2s_weight > 0 and self.ps2s_model is not None:
            self.log("train/ps2s_loss", ps2s_loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        neural, labels, neural_lens, label_lens, days = batch
        loss, logits, input_lengths, ps2s_loss = self._compute_loss(
            neural, labels, neural_lens, label_lens, days
        )
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.ps2s_weight > 0 and self.ps2s_model is not None:
            self.log("val/ps2s_loss", ps2s_loss, on_step=False, on_epoch=True)

        with torch.no_grad():
            pred = self._ctc_greedy_decode(logits, input_lengths)
            cer = self._token_error_rate(pred, labels, label_lens)
            self.val_cer(cer)
            self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)
            if not self._val_example_logged and pred:
                tgt_len = int(label_lens[0].item())
                tgt_seq = labels[0, :tgt_len].tolist()
                # uncorrected/noisy phonemes (CTC greedy before PS2S)
                noisy_ph = self._pred_tokens_to_phonemes(pred[0])
                pred_ph = self._pred_tokens_to_phonemes(pred[0])
                tgt_ph = [
                    self.phoneme_labels[t - 1] if self.phoneme_labels and 1 <= t <= len(self.phoneme_labels) else str(t)
                    for t in tgt_seq
                ]
                self.print(f"[val example] noisy: {' '.join(noisy_ph)} | pred: {' '.join(pred_ph)} | target: {' '.join(tgt_ph)}")
                self._val_example_logged = True

    def test_step(self, batch, batch_idx):
        neural, labels, neural_lens, label_lens, days = batch
        loss, logits, input_lengths, ps2s_loss = self._compute_loss(
            neural, labels, neural_lens, label_lens, days
        )
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.ps2s_weight > 0 and self.ps2s_model is not None:
            self.log("test/ps2s_loss", ps2s_loss, on_step=False, on_epoch=True)

        with torch.no_grad():
            pred = self._ctc_greedy_decode(logits, input_lengths)
            cer = self._token_error_rate(pred, labels, label_lens)
            self.test_cer(cer)
            self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage):
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    # make sure PS2S participates in optimization
    def configure_optimizers(self):
        if self.hparams.optimizer is None:
            return None

        # all trainable parameters, including PS2S
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

    # ensure PS2S follows train/eval modes
    def train(self, mode: bool = True):
        super().train(mode)
        if self.ps2s_model is not None:
            self.ps2s_model.train(mode)
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
    class DummyGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.strideLen = 4
            self.kernelLen = 16
            self.gru = torch.nn.GRU(256, 128, batch_first=True)
            self.fc = torch.nn.Linear(128, 41)

        def forward(self, x, days):
            h, _ = self.gru(x)
            return self.fc(h)

    dummy = DummyGRU()
    _ = SpeechModuleWithPS2S(
        net=dummy, optimizer=None, scheduler=None, compile=False
    )
    print("Sanity check OK.")
