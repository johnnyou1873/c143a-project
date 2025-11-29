import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

log = logging.getLogger(__name__)


def _tokens_to_prompt(tokens: Iterable[int], phoneme_map: Optional[List[str]] = None) -> str:
    token_str = " ".join(str(t) for t in tokens)
    mapping_block = ""
    if phoneme_map:
        mapping_lines = "\n".join(f"{idx}: {label}" for idx, label in enumerate(phoneme_map))
        mapping_block = (
            "Use this mapping from ids to phoneme symbols (0 is the CTC blank and is omitted):\n"
            f"{mapping_lines}\n"
        )
    return (
        "You are cleaning up a decoded phoneme sequence for an English speaker.\n"
        "Phonemes are represented as integers separated by spaces and do not include the blank token 0.\n"
        f"{mapping_block}"
        "Return only the corrected phoneme sequence as space-separated integers.\n"
        f"Decoded phonemes: {token_str}\n"
        "Corrected phonemes:"
    )


def _parse_tokens(text: str) -> List[int]:
    # prefer explicit integer lists; fallback to all integers in the string
    candidates = re.findall(r"-?\d+", text)
    return [int(c) for c in candidates]


class LocalLLMCorrector:
    """Optional local LLM client. Falls back to echoing the input if unavailable."""

    def __init__(
        self,
        model_name: str = "gpt-oss-20b",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        enabled: bool = False,
        phoneme_map: Optional[List[str]] = None,
    ) -> None:
        self.enabled = enabled
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.phoneme_map = phoneme_map or []
        self.model = None
        self.tokenizer = None

        if not self.enabled:
            log.info("LLM correction disabled; using greedy outputs directly.")
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            self.model.eval()
            log.info("Loaded local LLM '%s' for correction on device %s", model_name, device)
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.warning(
                "Could not load local LLM '%s'; falling back to identity corrections. Error: %s",
                model_name,
                exc,
            )
            self.enabled = False

    @torch.no_grad()
    def __call__(self, decoded_tokens: List[int]) -> List[int]:
        if not self.enabled or self.model is None or self.tokenizer is None:
            return decoded_tokens

        prompt = _tokens_to_prompt(decoded_tokens, self.phoneme_map if self.phoneme_map else None)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        do_sample = self.temperature > 0
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected = _parse_tokens(completion)
        return corrected if corrected else decoded_tokens


class LLMSpeechModule(LightningModule):
    """LightningModule with on-the-fly LLM-corrected CTC loss."""

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
        llm_weight: float = 0.3,
        llm_model_name: str = "gpt-oss-20b",
        llm_enabled: bool = False,
        llm_max_new_tokens: int = 64,
        llm_temperature: float = 0.0,
        phoneme_labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.llm_weight = float(llm_weight)
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_cer = MeanMetric()

        self.llm_corrector = LocalLLMCorrector(
            model_name=llm_model_name,
            max_new_tokens=llm_max_new_tokens,
            temperature=llm_temperature,
            enabled=llm_enabled,
            phoneme_map=phoneme_labels,
        )

    def forward(self, neural: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        return self.net(neural, days)

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_cer.reset()

    def _compute_input_lengths(self, neural_lens: torch.Tensor) -> torch.Tensor:
        stride = self.net.strideLen
        kernel = self.net.kernelLen
        lengths = (neural_lens - kernel) // stride + 1
        return torch.clamp(lengths, min=1)

    def _ctc_greedy_decode(
        self, logits: torch.Tensor, input_lengths: torch.Tensor
    ) -> List[List[int]]:
        preds = logits.argmax(dim=-1)
        sequences: List[List[int]] = []
        for seq, length in zip(preds, input_lengths):
            trimmed = seq[: int(length)]
            collapsed = torch.unique_consecutive(trimmed)
            decoded = [int(t.item()) for t in collapsed if int(t.item()) != 0]
            sequences.append(decoded)
        return sequences

    def _pack_labels(
        self, sequences: List[List[int]], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not sequences:
            return torch.zeros((0, 0), device=device, dtype=torch.int64), torch.zeros(
                (0,), device=device, dtype=torch.int64
            )

        max_len = max((len(seq) for seq in sequences), default=0)
        padded = torch.zeros((len(sequences), max_len), device=device, dtype=torch.int64)
        lengths = torch.zeros((len(sequences),), device=device, dtype=torch.int64)
        for idx, seq in enumerate(sequences):
            if not seq:
                continue
            end = min(len(seq), max_len)
            padded[idx, :end] = torch.tensor(seq[:end], device=device, dtype=torch.int64)
            lengths[idx] = end
        return padded, lengths

    def _token_error_rate(
        self, pred_tokens: List[List[int]], labels: torch.Tensor, label_lens: torch.Tensor
    ) -> float:
        total_dist = 0
        total_length = 0
        for pred, target, tgt_len in zip(pred_tokens, labels, label_lens):
            tgt_seq = target[: int(tgt_len)].tolist()
            total_dist += self._levenshtein(pred, tgt_seq)
            total_length += len(tgt_seq)
        if total_length == 0:
            return 0.0
        return float(total_dist) / float(total_length)

    @staticmethod
    def _levenshtein(source: List[int], target: List[int]) -> int:
        if len(source) < len(target):
            source, target = target, source

        previous_row = list(range(len(target) + 1))
        for i, src_token in enumerate(source, start=1):
            current_row = [i]
            for j, tgt_token in enumerate(target, start=1):
                insertions = previous_row[j] + 1
                deletions = current_row[j - 1] + 1
                substitutions = previous_row[j - 1] + (src_token != tgt_token)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _llm_correct(self, decoded: List[List[int]]) -> List[List[int]]:
        corrected: List[List[int]] = []
        for seq in decoded:
            corrected_seq = self.llm_corrector(seq)
            corrected.append(corrected_seq if corrected_seq else seq)
        return corrected

    def _compute_losses(
        self,
        neural: torch.Tensor,
        labels: torch.Tensor,
        neural_lens: torch.Tensor,
        label_lens: torch.Tensor,
        days: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]:
        logits = self.forward(neural, days)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        input_lengths = self._compute_input_lengths(neural_lens).to(logits.device)

        base_loss = self.ctc_loss(log_probs, labels, input_lengths, label_lens)

        decoded_tokens = self._ctc_greedy_decode(logits, input_lengths)
        corrected_tokens = self._llm_correct(decoded_tokens)
        corrected_labels, corrected_lens = self._pack_labels(corrected_tokens, logits.device)

        corrected_loss = self.ctc_loss(log_probs, corrected_labels, input_lengths, corrected_lens)

        weight = max(min(self.llm_weight, 1.0), 0.0)
        total_loss = (1.0 - weight) * base_loss + weight * corrected_loss

        return total_loss, base_loss, corrected_loss, logits, input_lengths, corrected_tokens

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        neural, labels, neural_lens, label_lens, days = batch
        total_loss, base_loss, corrected_loss, _, _, _ = self._compute_losses(
            neural, labels, neural_lens, label_lens, days
        )

        self.train_loss(total_loss)
        self.log("train/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_base", base_loss, on_step=False, on_epoch=True)
        self.log("train/loss_llm", corrected_loss, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        neural, labels, neural_lens, label_lens, days = batch
        total_loss, base_loss, corrected_loss, logits, input_lengths, corrected_tokens = (
            self._compute_losses(neural, labels, neural_lens, label_lens, days)
        )

        self.val_loss(total_loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_base", base_loss, on_step=False, on_epoch=True)
        self.log("val/loss_llm", corrected_loss, on_step=False, on_epoch=True)

        cer_value = self._token_error_rate(corrected_tokens, labels, label_lens)
        self.val_cer(cer_value)
        self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        neural, labels, neural_lens, label_lens, days = batch
        total_loss, base_loss, corrected_loss, logits, input_lengths, corrected_tokens = (
            self._compute_losses(neural, labels, neural_lens, label_lens, days)
        )

        self.test_loss(total_loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss_base", base_loss, on_step=False, on_epoch=True)
        self.log("test/loss_llm", corrected_loss, on_step=False, on_epoch=True)

        cer_value = self._token_error_rate(corrected_tokens, labels, label_lens)
        self.log("test/cer", cer_value, on_step=False, on_epoch=True, prog_bar=True)

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
    _ = LLMSpeechModule(None, None, None, None)
