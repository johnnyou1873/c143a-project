import torch
import torch.nn as nn

class CTCLossWithLabelSmoothing(nn.Module):
    """
    Wraps standard CTCLoss and adds label smoothing.

    log_probs should be log-softmax over classes, shape (T, N, C), same as torch.nn.CTCLoss.
    """
    def __init__(self, blank=0, reduction="mean", zero_infinity=True, label_smoothing=0.0):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.label_smoothing = label_smoothing
        self.base_ctc = nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Standard CTC loss (this corresponds to one-hot targets)
        base_loss = self.base_ctc(log_probs, targets, input_lengths, target_lengths)

        # No smoothing -> just CTC
        if self.label_smoothing <= 0.0:
            return base_loss

        # Label smoothing: mix base CTC loss with a uniform-target loss.
        # For a smoothed target q = (1-ε)*one_hot + ε*uniform,
        # CE(q, p) = (1-ε)*CE(one_hot, p) + ε*CE(uniform, p).
        #
        # We approximate CE(uniform, p) frame-wise using the log_probs:
        # CE(uniform, p) = - 1/K * sum_k log p_k
        # => -mean over class dimension of log_probs.
        #
        # log_probs: (T, N, C)
        uniform_ce_per_frame = -log_probs.mean(dim=-1)  # (T, N)
        uniform_ce = uniform_ce_per_frame.mean()        # scalar

        eps = self.label_smoothing
        smoothed_loss = (1.0 - eps) * base_loss + eps * uniform_ce
        return smoothed_loss
