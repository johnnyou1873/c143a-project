from typing import Any, Dict, List, Optional, Tuple
import os
from pathlib import Path
import functools

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from edit_distance import SequenceMatcher
import yaml


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
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.proj = torch.nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = src == self.pad_idx
        tgt_key_padding_mask = tgt_in == self.pad_idx
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            tgt_in.size(1), device=tgt_in.device
        )

        # Encoder can see current and limited future (lookahead tokens)
        src_seq_len = src.size(1)
        i = torch.arange(src_seq_len, device=src.device).unsqueeze(1)
        j = torch.arange(src_seq_len, device=src.device)
        src_mask = torch.where(j - i > self.lookahead, float("-inf"), 0.0)

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


class SpeechModuleWithPS2S(LightningModule):
    """LightningModule that combines a pretrained GRU decoder with a phoneme seq2seq model."""

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
            raise ValueError("`net` must be provided to SpeechModuleWithPS2S.")

        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

        # Load GRU weights if available
        self._load_gru_weights(gru_ckpt_path)

        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.ps2s_weight = ps2s_weight
        self.ps2s_max_len = ps2s_max_len

        self.ps2s_model, self.ps2s_meta = self._load_ps2s(ps2s_model_path)
        self.ps2s_loss = torch.nn.CrossEntropyLoss(ignore_index=self.ps2s_meta["pad_idx"])

        default_labels = self._load_default_phoneme_labels()
        self.phoneme_labels = phoneme_labels or default_labels

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_cer = MeanMetric()
        self.test_cer = MeanMetric()

    # ------------------------- utility loaders ------------------------- #
    def _load_gru_weights(self, ckpt_path: str) -> None:
        path = Path(ckpt_path)
        if not path.exists():
            return
        try:
            torch.serialization.add_safe_globals([functools.partial])
        except Exception:
            pass
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            filtered = {k.replace("net.", ""): v for k, v in state_dict.items() if k.startswith("net.")}
            missing, unexpected = self.net.load_state_dict(filtered, strict=False)
            if missing:
                self.print(f"[ps2s] GRU missing keys: {missing}")
            if unexpected:
                self.print(f"[ps2s] GRU unexpected keys: {unexpected}")
        except Exception as e:
            self.print(f"[ps2s] Failed to load GRU weights: {e}")

    def _load_ps2s(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        meta = {
            "phoneme_to_idx": ckpt["phoneme_to_idx"],
            "idx_to_phoneme": ckpt["idx_to_phoneme"],
            "pad_idx": int(ckpt["pad_idx"]),
            "sos_idx": int(ckpt["sos_idx"]),
            "eos_idx": int(ckpt["eos_idx"]),
            "max_len": int(ckpt["max_len"]),
        }
        model_kwargs = ckpt["model_kwargs"]
        ps2s = Seq2SeqCorrection(pad_idx=meta["pad_idx"], lookahead=1, **model_kwargs)
        ps2s.load_state_dict(ckpt["state_dict"], strict=True)
        return ps2s, meta

    def _load_default_phoneme_labels(self) -> Optional[List[str]]:
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

    # ------------------------- helpers ------------------------- #
    def _compute_input_lengths(self, neural_lens: torch.Tensor) -> torch.Tensor:
        stride = self.net.strideLen
        kernel = self.net.kernelLen
        lengths = torch.div(neural_lens - kernel, stride, rounding_mode="floor")
        return lengths.to(torch.int32)

    def _ctc_greedy_decode(
        self, logits: torch.Tensor, input_lengths: torch.Tensor
    ) -> List[List[int]]:
        preds = logits.argmax(dim=-1)
        sequences: List[List[int]] = []
        for seq, length in zip(preds, input_lengths):
            trimmed = seq[: int(length)]
            collapsed = torch.unique_consecutive(trimmed)
            decoded = [int(t) for t in collapsed.tolist() if int(t) != 0]
            sequences.append(decoded)
        return sequences

    def _token_error_rate(
        self, pred_tokens: List[List[int]], labels: torch.Tensor, label_lens: torch.Tensor
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

    def _labels_to_phonemes(self, labels: torch.Tensor, label_lens: torch.Tensor) -> List[List[str]]:
        phs: List[List[str]] = []
        for seq, length in zip(labels, label_lens):
            trimmed = seq[: int(length)].tolist()
            if self.phoneme_labels:
                phs.append([self.phoneme_labels[int(t) - 1] if int(t) > 0 and int(t) - 1 < len(self.phoneme_labels) else "<unk>" for t in trimmed])
            else:
                phs.append([str(int(t)) for t in trimmed])
        return phs

    def _phonemes_to_ps2s_tokens(self, phonemes: List[str]) -> List[int]:
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
        return tgt_in, tgt_out

    def _prepare_ps2s_batch(
        self, labels: torch.Tensor, label_lens: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ph_lists = self._labels_to_phonemes(labels, label_lens)
        src_tokens, tgt_in, tgt_out = [], [], []
        pad = self.ps2s_meta["pad_idx"]
        for phs in ph_lists:
            inp, out = self._phonemes_to_ps2s_tokens(phs)
            src_tokens.append(inp)
            tgt_in.append(inp)
            tgt_out.append(out)
        src_t = torch.tensor(src_tokens, device=device, dtype=torch.long)
        tgt_in_t = torch.tensor(tgt_in, device=device, dtype=torch.long)
        tgt_out_t = torch.tensor(tgt_out, device=device, dtype=torch.long)
        return src_t, tgt_in_t, tgt_out_t

    # ------------------------- forward & loss ------------------------- #
    def forward(self, neural: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        return self.net(neural, days)

    def _compute_loss(
        self,
        neural: torch.Tensor,
        labels: torch.Tensor,
        neural_lens: torch.Tensor,
        label_lens: torch.Tensor,
        days: torch.Tensor,
    ):
        logits = self.forward(neural, days)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        input_lengths = self._compute_input_lengths(neural_lens).to(logits.device)
        label_lens = label_lens.to(logits.device)
        ctc = self.ctc_loss(log_probs, labels, input_lengths, label_lens)

        ps2s_loss = torch.zeros((), device=logits.device)
        if self.ps2s_weight > 0:
            self.ps2s_model = self.ps2s_model.to(logits.device)
            src, tgt_in, tgt_out = self._prepare_ps2s_batch(labels, label_lens, logits.device)
            ps2s_logits = self.ps2s_model(src, tgt_in)
            ps2s_loss = self.ps2s_loss(ps2s_logits.reshape(-1, ps2s_logits.size(-1)), tgt_out.reshape(-1))

        total_loss = ctc + self.ps2s_weight * ps2s_loss
        return total_loss, logits, input_lengths, ps2s_loss

    # ------------------------- lightning hooks ------------------------- #
    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_cer.reset()
        self.test_cer.reset()

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        neural, labels, neural_lens, label_lens, days = batch
        loss, _, _, ps2s_loss = self._compute_loss(neural, labels, neural_lens, label_lens, days)
        self.train_loss(loss)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.ps2s_weight > 0:
            self.log("train/ps2s_loss", ps2s_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        neural, labels, neural_lens, label_lens, days = batch
        loss, logits, input_lengths, ps2s_loss = self._compute_loss(
            neural, labels, neural_lens, label_lens, days
        )
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.ps2s_weight > 0:
            self.log("val/ps2s_loss", ps2s_loss, on_step=False, on_epoch=True, prog_bar=False)
        with torch.no_grad():
            pred_tokens = self._ctc_greedy_decode(logits, input_lengths)
            cer_value = self._token_error_rate(pred_tokens, labels, label_lens)
            self.val_cer(cer_value)
            self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        neural, labels, neural_lens, label_lens, days = batch
        loss, logits, input_lengths, ps2s_loss = self._compute_loss(
            neural, labels, neural_lens, label_lens, days
        )
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.ps2s_weight > 0:
            self.log("test/ps2s_loss", ps2s_loss, on_step=False, on_epoch=True, prog_bar=False)
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
    _ = SpeechModuleWithPS2S(net=None, optimizer=None, scheduler=None, compile=False)
