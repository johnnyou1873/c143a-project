from typing import Any, Dict, List, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from edit_distance import SequenceMatcher


class SpeechModule(LightningModule):
    """`LightningModule` for BMI speech task.

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
        """Forward pass + CTC loss."""
        logits = self.forward(neural, days)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        input_lengths = self._compute_input_lengths(neural_lens).to(logits.device)
        loss = self.ctc_loss(log_probs, labels, input_lengths, label_lens)
        return loss, logits, input_lengths

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
            cer_value = self._token_error_rate(pred_tokens, labels, label_lens)
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
            cer_value = self._token_error_rate(pred_tokens, labels, label_lens)
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


if __name__ == "__main__":
    _ = SpeechModule(None, None, None, None)
