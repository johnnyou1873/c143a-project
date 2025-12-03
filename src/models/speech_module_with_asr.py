from typing import Any, Dict, List, Tuple

import torch
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
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
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
        asr_model_id: str = "nvidia/parakeet-tdt-0.6b-v2",
        freeze_asr: bool = True,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        # Load NeMo ASR model
        self.asr_ctc = nemo_asr.models.ASRModel.from_pretrained(model_name=asr_model_id)
        if freeze_asr:
            for p in self.asr_ctc.parameters():
                p.requires_grad = False
            self.asr_ctc.eval()
        else:
            self.asr_ctc.train()

        cfg = getattr(self.asr_ctc, "cfg", None)
        feat_in = 1024
        if cfg is not None and hasattr(cfg, "encoder") and hasattr(cfg.encoder, "feat_in"):
            feat_in = cfg.encoder.feat_in
        asr_hidden = getattr(cfg.encoder, "d_model", getattr(cfg.encoder, "hidden_size", feat_in)) if cfg and hasattr(cfg, "encoder") else feat_in
        parakeet_vocab = getattr(cfg.decoder, "num_classes", n_classes + 1) if cfg and hasattr(cfg, "decoder") else n_classes + 1

        gru_dim = n_units * (2 if bidirectional else 1)
        self.gru_to_asr = torch.nn.Linear(gru_dim, feat_in)
        self.asr_to_phoneme = torch.nn.Linear(parakeet_vocab, n_classes)

        # helper: keep only A-Z characters
        self._letters = set(string.ascii_letters)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_cer = MeanMetric()
        self.test_cer = MeanMetric()

    def forward(self, neural: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        return self.net(neural, days)

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_cer.reset()
        self.test_cer.reset()

    def _compute_input_lengths(self, neural_lens: torch.Tensor) -> torch.Tensor:
        stride = self.net.strideLen
        kernel = self.net.kernelLen
        lengths = torch.div(neural_lens - kernel, stride, rounding_mode="floor")
        return lengths.to(torch.int32)

    def _ctc_greedy_decode(self, logits: torch.Tensor, input_lengths: torch.Tensor) -> List[List[int]]:
        preds = logits.argmax(dim=-1)
        sequences: List[List[int]] = []
        for seq, length in zip(preds, input_lengths):
            trimmed = seq[: int(length)]
            collapsed = torch.unique_consecutive(trimmed)
            decoded = [int(t) for t in collapsed.tolist() if int(t) != 0]
            sequences.append(decoded)
        return sequences

    def _token_error_rate(self, pred_tokens: List[List[int]], labels: torch.Tensor, label_lens: torch.Tensor) -> float:
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

    def _compute_loss(
        self,
        neural: torch.Tensor,
        labels: torch.Tensor,
        neural_lens: torch.Tensor,
        label_lens: torch.Tensor,
        days: torch.Tensor,
    ):
        # 1) GRU encoder on spike data
        gru_out = self.net(neural, days)             # (B, T', H)

        # 2) Map GRU output into ASR encoder feature space (feat_in)
        enc_in = self.gru_to_asr(gru_out)            # (B, T', feat_in)
        enc_lengths = self._compute_input_lengths(neural_lens).to(self.device)

        # 3) Run through ASR encoder (bypass preprocessor)
        enc_out, enc_out_len = self.asr_ctc.encoder(
            audio_signal=enc_in,
            length=enc_lengths,
            bypass_pre_encode=True,
        )

        # 4) ASR CTC head (subword logits)
        if hasattr(self.asr_ctc, "ctc_lin"):
            parakeet_logits = self.asr_ctc.ctc_lin(enc_out)
        else:
            if not hasattr(self, "parakeet_head"):
                self.parakeet_head = torch.nn.Linear(enc_out.size(-1), self.asr_to_phoneme.in_features)
            if self.parakeet_head.in_features != enc_out.size(-1):
                self.parakeet_head = torch.nn.Linear(enc_out.size(-1), self.asr_to_phoneme.in_features)
            self.parakeet_head = self.parakeet_head.to(enc_out.device)
            parakeet_logits = self.parakeet_head(enc_out)

        # 5) Project to BMI phonemes
        logits = self.asr_to_phoneme(parakeet_logits)

        # 6) CTC loss
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        loss = self.ctc_loss(log_probs, labels, enc_out_len, label_lens)
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
    _ = SpeechModule(None, None, None, None)
