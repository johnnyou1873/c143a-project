from typing import Any, Dict, List, Optional, Tuple
import os

import torch
torch.set_float32_matmul_precision("medium")
import nemo.collections.asr as nemo_asr
from lightning import LightningModule
from torchmetrics import MeanMetric
from edit_distance import SequenceMatcher
import string


class SpeechModule(LightningModule):
    """LightningModule for BMI speech with a pretrained NeMo ASR backend."""

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
        # n_classes: backward-compatible alias for n_phonemes (excludes CTC blank).
        n_classes: int = 40,
        # n_phonemes: size of *phoneme* vocabulary (excluding CTC blank).
        # The actual CTC vocab size will be n_phonemes + 1, with blank at index 0.
        n_phonemes: Optional[int] = None,
        n_input_features: int = 256,
        dropout: float = 0.4,
        gaussian_smooth_width: float = 2.0,
        stride_len: int = 4,
        kernel_len: int = 32,
        bidirectional: bool = False,
        l2_decay: float = 1e-5,
        seed: int = 0,
        asr_model_id: str = "nvidia/parakeet-tdt-0.6b-v2",
        freeze_asr: bool = True,
    ) -> None:
        super().__init__()

        # store hyperparameters (excluding the actual `net` object)
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # CTC vocabulary size = blank (0) + phonemes (1..n_phonemes)
        # Hydra config passes `n_classes`; fall back to it when n_phonemes isn't provided.
        self.n_phonemes = n_phonemes if n_phonemes is not None else n_classes
        self.vocab_size = self.n_phonemes + 1  # includes blank

        self.ctc_loss = torch.nn.CTCLoss(
            blank=0,          # we reserve index 0 for blank
            reduction="mean",
            zero_infinity=True,
        )

        # Load NeMo ASR model (either by name or from a local checkpoint path)
        if os.path.exists(asr_model_id):
            # Local .nemo file
            self.asr_ctc = nemo_asr.models.ASRModel.restore_from(restore_path=asr_model_id)
        else:
            # Pretrained by model name (NGC / HF)
            self.asr_ctc = nemo_asr.models.ASRModel.from_pretrained(model_name=asr_model_id)

        if freeze_asr:
            for p in self.asr_ctc.parameters():
                p.requires_grad = False
            self.asr_ctc.eval()
        else:
            self.asr_ctc.train()

        cfg = getattr(self.asr_ctc, "cfg", None)

        # Parakeet encoder expects 1024-dim features when bypassing preprocessor
        # (you already know this from the ASR side).
        feat_in = 1024

        # Number of units in the ASR decoder (subword vocab)
        if cfg is not None and hasattr(cfg, "decoder") and hasattr(cfg.decoder, "num_classes"):
            parakeet_vocab = cfg.decoder.num_classes
        else:
            # Fallback: use CTC vocab size
            parakeet_vocab = self.vocab_size

        # Align input dim with GRU decoder logits when available
        if hasattr(self.net, "fc_decoder_out"):
            gru_dim = self.net.fc_decoder_out.out_features
        else:
            gru_dim = n_units * (2 if bidirectional else 1)

        # Map BMI GRU output -> Parakeet encoder input
        self.gru_to_asr = torch.nn.Linear(gru_dim, feat_in)

        # Project Parakeet token logits -> BMI phoneme CTC vocab (including blank)
        self.asr_to_phoneme = torch.nn.Linear(parakeet_vocab, self.vocab_size)

        # helper: keep only A-Z characters (currently unused but kept for later use)
        self._letters = set(string.ascii_letters)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_cer = MeanMetric()
        self.test_cer = MeanMetric()

    def forward(self, neural: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        """Forward for compatibility; just uses the BMI net."""
        return self.net(neural, days)

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_cer.reset()
        self.test_cer.reset()

    def _compute_input_lengths(self, neural_lens: torch.Tensor) -> torch.Tensor:
        """Approximate GRU / ASR input length from raw neural length using stride & kernel."""
        stride = self.net.strideLen
        kernel = self.net.kernelLen
        # make sure we don't go negative
        effective = torch.clamp(neural_lens - kernel, min=0)
        lengths = torch.div(effective, stride, rounding_mode="floor")
        return lengths.to(torch.int32)

    def _ctc_greedy_decode(
        self,
        logits: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> List[List[int]]:
        """Simple greedy decode with collapse + blank removal."""
        preds = logits.argmax(dim=-1)  # (B, T)
        sequences: List[List[int]] = []
        for seq, length in zip(preds, input_lengths):
            trimmed = seq[: int(length)]
            collapsed = torch.unique_consecutive(trimmed)
            # drop blank index 0
            decoded = [int(t) for t in collapsed.tolist() if int(t) != 0]
            sequences.append(decoded)
        return sequences

    def _token_error_rate(
        self,
        pred_tokens: List[List[int]],
        labels: torch.Tensor,
        label_lens: torch.Tensor,
    ) -> float:
        total_edit_distance = 0
        total_seq_length = 0
        for pred, target, tgt_len in zip(pred_tokens, labels, label_lens):
            true_seq = target[: int(tgt_len)].tolist()
            matcher = SequenceMatcher(a=true_seq, b=pred)
            total_edit_distance += matcher.distance()
            total_seq_length += len(true_seq)
        if total_seq_length == 0:
            return 0.0
        return total_edit_distance / total_seq_length

    def _encode_with_asr(
        self,
        enc_in: torch.Tensor,
        enc_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the NeMo ASR encoder on pre-projected features.
        We try a 'bypass_pre_encode' call first, then fall back to a simpler signature.
        """
        # Ensure shapes are what the encoder expects.
        # If Parakeet uses (B, T, D) this is fine; if it uses (B, D, T) you may need to transpose here.
        try:
            enc_out, enc_out_len = self.asr_ctc.encoder(
                audio_signal=enc_in,
                length=enc_lengths,
                bypass_pre_encode=True,
            )
        except TypeError:
            # Fallback: some encoders only accept positional arguments or lack bypass_pre_encode
            enc_out, enc_out_len = self.asr_ctc.encoder(enc_in, enc_lengths)

        return enc_out, enc_out_len

    def _compute_loss(
        self,
        neural: torch.Tensor,
        labels: torch.Tensor,
        neural_lens: torch.Tensor,
        label_lens: torch.Tensor,
        days: torch.Tensor,
    ):
        # 1) GRU encoder on spike data -> (B, T', H)
        gru_out = self.net(neural, days)

        # 2) Map GRU output into ASR encoder feature space (feat_in)
        enc_in = self.gru_to_asr(gru_out)  # (B, T', feat_in)
        enc_lengths = self._compute_input_lengths(neural_lens).to(enc_in.device)

        # 3) Run through ASR encoder (bypass preprocessor where supported)
        enc_out, enc_out_len = self._encode_with_asr(enc_in, enc_lengths)

        # 4) ASR head: project encoder hidden states -> ASR vocab (subword logits)
        if hasattr(self.asr_ctc, "ctc_lin"):
            parakeet_logits = self.asr_ctc.ctc_lin(enc_out)
        else:
            # Dynamically create a simple linear head if the model doesn't expose a CTC head.
            if not hasattr(self, "parakeet_head") or self.parakeet_head.in_features != enc_out.size(-1):
                self.parakeet_head = torch.nn.Linear(
                    in_features=enc_out.size(-1),
                    out_features=self.asr_to_phoneme.in_features,
                ).to(enc_out.device)
            parakeet_logits = self.parakeet_head(enc_out)

        # 5) Project to BMI phoneme CTC vocab (includes blank)
        logits = self.asr_to_phoneme(parakeet_logits)  # (B, T_asr, vocab_size)

        # 6) CTC loss
        # CTC expects (T, N, C)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

        # Ensure integer lengths on the correct device
        enc_out_len = enc_out_len.to(device=log_probs.device, dtype=torch.int32)
        label_lens = label_lens.to(device=log_probs.device, dtype=torch.int32)

        loss = self.ctc_loss(
            log_probs,
            labels,
            enc_out_len,
            label_lens,
        )
        return loss, logits, enc_out_len

    def model_step(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        neural, labels, neural_lens, label_lens, days = batch
        loss, _, _ = self._compute_loss(neural, labels, neural_lens, label_lens, days)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        neural, labels, neural_lens, label_lens, days = batch
        loss, _, _ = self._compute_loss(neural, labels, neural_lens, label_lens, days)
        self.train_loss(loss)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        neural, labels, neural_lens, label_lens, days = batch
        loss, logits, input_lengths = self._compute_loss(neural, labels, neural_lens, label_lens, days)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        with torch.no_grad():
            pred_tokens = self._ctc_greedy_decode(logits, input_lengths)
            cer_value = self._token_error_rate(pred_tokens, labels, label_lens)
            self.val_cer(cer_value)
            self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)
            # Show one example prediction vs. ground truth for quick debugging.
            if pred_tokens:
                target_len = int(label_lens[0].item())
                target_seq = labels[0, :target_len].detach().cpu().tolist()
                pred_seq = pred_tokens[0]
                self.print(f"[val example] pred={pred_seq} target={target_seq}")

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        neural, labels, neural_lens, label_lens, days = batch
        loss, logits, input_lengths = self._compute_loss(neural, labels, neural_lens, label_lens, days)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        with torch.no_grad():
            pred_tokens = self._ctc_greedy_decode(logits, input_lengths)
            cer_value = self._token_error_rate(pred_tokens, labels, label_lens)
            self.test_cer(cer_value)
            self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        # `optimizer` and `scheduler` are expected to be callables from the config
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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


if __name__ == "__main__":
    # This is just a construction sanity check; in practice Hydra will pass real `net` and optimizers.
    _ = SpeechModule(net=None, optimizer=None, scheduler=None, compile=False)
