from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from edit_distance import SequenceMatcher


class LLMCorrector:
    """Lightweight wrapper to optionally call an external LLM for phoneme corrections.

    This is written defensively: if the runtime doesn't have an LLM client available,
    it will simply return the original predicted tokens so training still runs.
    """

    def __init__(
        self,
        enabled: bool,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        prompt: str,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.prompt = prompt
        self._client = self._maybe_init_model(enabled)

    def _maybe_init_model(self, enabled: bool) -> Optional[Any]:
        if not enabled:
            return None
        try:
            import ollama  # type: ignore

            # lazy client; server availability is checked on first request
            return ollama
        except Exception:
            return None

    def correct_tokens(
        self, phoneme_labels: List[str], predicted_tokens: List[List[int]]
    ) -> List[List[int]]:
        if not self.enabled or self._client is None:
            return predicted_tokens

        allowed_list = ", ".join(phoneme_labels)
        system_parts = []
        if self.prompt:
            system_parts.append(self.prompt)
        system_parts.append("You may only output phonemes from the allowed set.")
        system_parts.append(f"Allowed phonemes: {allowed_list}")
        system_content = "\n".join(system_parts).strip()

        corrected: List[List[int]] = []
        for tokens in predicted_tokens:
            user_prompt = self._build_prompt(phoneme_labels, tokens)
            try:
                response = self._client.chat(  # type: ignore[attr-defined]
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={
                        "temperature": max(self.temperature, 0.0),
                        "num_predict": self.max_new_tokens,
                    },
                )
                text = response.get("message", {}).get("content", "") if response else ""
                parsed = self._parse_phonemes(phoneme_labels, text)
                corrected.append(parsed if parsed else tokens)
            except Exception:
                corrected.append(tokens)
        return corrected

    def _build_prompt(self, phoneme_labels: List[str], tokens: List[int]) -> str:
        phonemes = [phoneme_labels[t] for t in tokens if 0 <= t < len(phoneme_labels)]
        sequence = " ".join(phonemes)
        return (
            f"Predicted phoneme sequence (space-separated): {sequence}\n"
            "Return a corrected phoneme sequence as a single line of space-separated phonemes. "
            "Do not explain your answer."
        )

    @staticmethod
    def _parse_phonemes(phoneme_labels: List[str], text: str) -> List[int]:
        labels_lower = {p.lower(): idx for idx, p in enumerate(phoneme_labels)}
        cleaned = (
            text.replace(",", " ")
            .replace("\n", " ")
            .replace("\r", " ")
            .strip()
        )
        tokens: List[int] = []
        for piece in cleaned.split():
            idx = labels_lower.get(piece.lower())
            if idx is not None:
                tokens.append(idx)
        return tokens


class LLMSpeechModule(LightningModule):
    """`LightningModule` for BMI speech task that includes extra linear layer and LLM for phoneme corrections.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

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
        # LLM guidance
        llm_enabled: bool = False,
        llm_weight: float = 0.3,
        llm_model_name: str = "gemma3:4b",
        llm_max_new_tokens: int = 64,
        llm_temperature: float = 0.0,
        phoneme_labels: Optional[List[str]] = None,
        llm_prompt: str = (
            "You are assisting with phoneme recognition. "
            "Given a rough phoneme prediction, return a cleaned list of phonemes "
            "using only the provided set."
        ),
    ) -> None:
        """Initialize a `SpeechModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.phoneme_labels = phoneme_labels or []
        self.llm_weight = llm_weight
        self.llm_corrector = LLMCorrector(
            enabled=llm_enabled,
            model_name=llm_model_name,
            max_new_tokens=llm_max_new_tokens,
            temperature=llm_temperature,
            prompt=llm_prompt,
        )
        self._llm_smoothing = 1e-4
        self._logged_val_example = False
        self._logged_test_example = False

        # track average losses
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_cer = MeanMetric()
        self.test_cer = MeanMetric()

    def forward(self, neural: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network.

        :param neural: Batched neural input of shape (batch, time, channels).
        :param days: Day indices per sample.
        :return: Sequence logits of shape (batch, time', classes).
        """
        return self.net(neural, days)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_cer.reset()
        self.test_cer.reset()
        self._logged_val_example = False
        self._logged_test_example = False

    def on_validation_epoch_start(self) -> None:
        self._logged_val_example = False

    def on_test_epoch_start(self) -> None:
        self._logged_test_example = False

    def _compute_input_lengths(self, neural_lens: torch.Tensor) -> torch.Tensor:
        """Compute GRU input lengths after striding."""
        stride = self.net.strideLen
        kernel = self.net.kernelLen
        lengths = torch.div(neural_lens - kernel, stride, rounding_mode="floor")
        return lengths.to(torch.int32)

    def _ctc_greedy_decode(
        self, logits: torch.Tensor, input_lengths: torch.Tensor
    ) -> List[List[int]]:
        """Greedy CTC decode with collapse + blank removal."""
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
        """Compute CER via edit distance on token sequences."""
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass + CTC loss with optional LLM-guided correction."""
        logits = self.forward(neural, days)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        input_lengths = self._compute_input_lengths(neural_lens).to(logits.device)
        base_loss = self.ctc_loss(log_probs, labels, input_lengths, label_lens)

        if self.phoneme_labels and self.llm_corrector.enabled:
            with torch.no_grad():
                pred_tokens = self._ctc_greedy_decode(logits, input_lengths)
                # Only send model predictions to the LLM; ground truth labels are never exposed.
                corrected_tokens = self.llm_corrector.correct_tokens(
                    self.phoneme_labels, pred_tokens
                )
            llm_log_probs = self._llm_log_probs_from_tokens(
                corrected_tokens,
                logits.shape[1],
                logits.shape[2],
                input_lengths,
                logits.device,
            )
            llm_loss = self.ctc_loss(llm_log_probs.transpose(0, 1), labels, input_lengths, label_lens)
            # Weighted combination of base CTC loss and LLM-corrected loss.
            loss = (1 - self.llm_weight) * base_loss + self.llm_weight * llm_loss
        else:
            llm_loss = None
            loss = base_loss

        return loss, logits, input_lengths

    def _llm_log_probs_from_tokens(
        self,
        corrected_tokens: List[List[int]],
        max_time: int,
        n_classes: int,
        input_lengths: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert LLM-corrected token sequences into log-probabilities over time.

        We create a sharp distribution for the suggested token at each timestep and
        default to a uniform-ish prior elsewhere. This stays fully differentiable
        and CTCLoss-compatible.
        """
        batch = len(corrected_tokens)
        log_probs = torch.full(
            (batch, max_time, n_classes),
            fill_value=1.0 / n_classes,
            device=device,
        ).log_()
        eps = self._llm_smoothing
        for i, tokens in enumerate(corrected_tokens):
            steps = int(input_lengths[i].item())
            if steps <= 0:
                continue
            seq = tokens if tokens else [0]
            for t in range(steps):
                idx = seq[min(int(t * len(seq) / max(1, steps)), len(seq) - 1)]
                idx = max(0, min(idx, n_classes - 1))
                probs = torch.full((n_classes,), eps, device=device)
                probs[idx] = 1.0 - eps * (n_classes - 1)
                log_probs[i, t] = torch.log(probs)
        return log_probs

    def model_step(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Shared forward-loss computation used by evaluation hooks."""
        neural, labels, neural_lens, label_lens, days = batch
        loss, _, _ = self._compute_loss(neural, labels, neural_lens, label_lens, days)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        neural, labels, neural_lens, label_lens, days = batch
        loss, _, _ = self._compute_loss(neural, labels, neural_lens, label_lens, days)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        neural, labels, neural_lens, label_lens, days = batch
        loss, logits, input_lengths = self._compute_loss(
            neural, labels, neural_lens, label_lens, days
        )

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # compute and log character error rate (greedy CTC decode)
        with torch.no_grad():
            pred_tokens = self._ctc_greedy_decode(logits, input_lengths)
            corrected_tokens = (
                self.llm_corrector.correct_tokens(self.phoneme_labels, pred_tokens)
                if self.llm_corrector.enabled and self.phoneme_labels
                else pred_tokens
            )
            self._maybe_log_llm_example("val", pred_tokens, corrected_tokens)
            cer_value = self._token_error_rate(corrected_tokens, labels, label_lens)
            self.val_cer(cer_value)
            self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        neural, labels, neural_lens, label_lens, days = batch
        loss, logits, input_lengths = self._compute_loss(
            neural, labels, neural_lens, label_lens, days
        )
        
        #loss = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # update and log metrics
        with torch.no_grad():
            pred_tokens = self._ctc_greedy_decode(logits, input_lengths)
            corrected_tokens = (
                self.llm_corrector.correct_tokens(self.phoneme_labels, pred_tokens)
                if self.llm_corrector.enabled and self.phoneme_labels
                else pred_tokens
            )
            self._maybe_log_llm_example("test", pred_tokens, corrected_tokens)
            cer_value = self._token_error_rate(corrected_tokens, labels, label_lens)
            self.test_cer(cer_value)
            self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
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

    def _tokens_to_phonemes(self, tokens: List[int]) -> str:
        """Convert integer tokens to a space-separated phoneme string."""
        return " ".join(
            self.phoneme_labels[t] for t in tokens if 0 <= t < len(self.phoneme_labels)
        )

    def _maybe_log_llm_example(
        self, stage: str, pred_tokens: List[List[int]], corrected_tokens: List[List[int]]
    ) -> None:
        """Log one example of LLM correction per validation/test epoch."""
        if not pred_tokens or not self.phoneme_labels:
            return
        if stage == "val" and self._logged_val_example:
            return
        if stage == "test" and self._logged_test_example:
            return

        idx = 0
        base_seq = self._tokens_to_phonemes(pred_tokens[idx])
        corr_seq = self._tokens_to_phonemes(corrected_tokens[idx])
        self.print(
            f"[{stage}] LLM correction example | predicted: '{base_seq}' | corrected: '{corr_seq}'"
        )

        if stage == "val":
            self._logged_val_example = True
        elif stage == "test":
            self._logged_test_example = True


if __name__ == "__main__":
    _ = LLMSpeechModule(None, None, None, None)
