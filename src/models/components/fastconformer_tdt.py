from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from ...utils.augmentations import GaussianSmoothing


def _causal_mask(
    seq_len: int, device: torch.device, lookahead: int = 0, left_context: Optional[int] = None
) -> torch.Tensor:
    """Build an additive attention mask that enforces causal / streaming behavior.

    Allowed positions are within [i - left_context, i + lookahead]; everything else is -inf.
    """
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)

    allowed = j <= i + lookahead
    if left_context is not None:
        allowed &= j >= i - left_context

    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = mask.masked_fill(allowed, 0.0)
    return mask


class FeedForwardModule(nn.Module):
    """Lightweight FFN used inside the Conformer blocks."""

    def __init__(self, d_model: int, expansion: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(d_model * expansion)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StreamingSelfAttention(nn.Module):
    """Multi-head attention with a built-in streaming (causal) mask."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        lookahead: int = 0,
        left_context: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.lookahead = lookahead
        self.left_context = left_context
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        attn_mask = _causal_mask(
            seq_len, device=x.device, lookahead=self.lookahead, left_context=self.left_context
        )
        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return out


class ConvModule(nn.Module):
    """Depthwise separable conv block with causal padding (FastConformer style)."""

    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1, padding=0)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, groups=d_model, padding=0
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, padding=0)
        self.dropout = dropout
        self.kernel_size = kernel_size

    def _causal_depthwise_conv(self, x: torch.Tensor) -> torch.Tensor:
        # causal left padding only
        if self.kernel_size > 1:
            padding = (self.kernel_size - 1, 0)
            x = F.pad(x, pad=padding)
        return self.depthwise_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = self._causal_depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)  # (B, T, D)
        return x + residual


class FastConformerBlock(nn.Module):
    """FastConformer encoder block with streaming self-attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_expansion: float = 4.0,
        conv_kernel: int = 15,
        dropout: float = 0.1,
        attn_lookahead: int = 0,
        attn_left_context: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn1 = FeedForwardModule(d_model, expansion=ff_expansion, dropout=dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = StreamingSelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            lookahead=attn_lookahead,
            left_context=attn_left_context,
        )
        self.conv = ConvModule(d_model=d_model, kernel_size=conv_kernel, dropout=dropout)
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2 = FeedForwardModule(d_model, expansion=ff_expansion, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))
        x = x + F.dropout(self.self_attn(self.self_attn_norm(x)), p=self.dropout, training=self.training)
        x = self.conv(x)
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        return self.final_norm(x)


class FastConformerTDT(nn.Module):
    """FastConformer-TDT encoder with causal masking for streaming decoding."""

    def __init__(
        self,
        neural_dim: int,
        n_classes: int,
        d_model: int = 256,
        n_layers: int = 8,
        nhead: int = 4,
        ff_expansion: float = 4.0,
        conv_kernel: int = 15,
        dropout: float = 0.1,
        strideLen: int = 4,
        kernelLen: int = 15,
        nDays: int = 24,
        gaussianSmoothWidth: float = 0.0,
        attn_lookahead: int = 0,
        attn_left_context: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth

        self.inputLayerNonlinearity = nn.Softsign()
        self.gaussianSmoother = (
            GaussianSmoothing(neural_dim, 20, gaussianSmoothWidth, dim=1)
            if gaussianSmoothWidth > 0
            else nn.Identity()
        )

        self.day_scale = nn.Parameter(torch.ones(nDays, neural_dim))
        self.day_bias = nn.Parameter(torch.zeros(nDays, neural_dim))

        self.subsample = nn.Conv1d(
            neural_dim, d_model, kernel_size=kernelLen, stride=strideLen, padding=0
        )
        self.subsample_norm = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList(
            [
                FastConformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    ff_expansion=ff_expansion,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                    attn_lookahead=attn_lookahead,
                    attn_left_context=attn_left_context,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput: torch.Tensor, dayIdx: torch.Tensor) -> torch.Tensor:
        # neuralInput: (B, T, C)
        x = neuralInput.transpose(1, 2)  # (B, C, T)
        x = self.gaussianSmoother(x)
        x = x.transpose(1, 2)  # (B, T, C)

        # day-specific affine transform
        scale = torch.index_select(self.day_scale, 0, dayIdx).unsqueeze(1)
        bias = torch.index_select(self.day_bias, 0, dayIdx).unsqueeze(1)
        x = self.inputLayerNonlinearity(x * scale + bias)

        # temporal subsampling + projection to model dimension
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.subsample(x)  # (B, d_model, T')
        x = x.transpose(1, 2)  # (B, T', d_model)
        x = self.subsample_norm(x)

        # encoder blocks
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.fc_out(x)
