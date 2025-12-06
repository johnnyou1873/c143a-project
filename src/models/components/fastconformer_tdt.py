from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from ...utils.augmentations import GaussianSmoothing
from .gru_diphone import DiphoneCodec, PHONEME_LABELS


# ============================================================
#                MASK (CACHED + DETACHED)
# ============================================================

def _causal_mask_cached(seq_len, device, lookahead, left_context, cache):
    key = (seq_len, device)
    if key in cache:
        return cache[key]

    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)

    allowed = j <= i + lookahead
    if left_context is not None:
        allowed &= j >= i - left_context

    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = mask.masked_fill(allowed, 0.0)

    # Important: do NOT track gradients
    mask = mask.detach()

    cache[key] = mask
    return mask


# ============================================================
#                     FEEDFORWARD MODULE
# ============================================================

class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, expansion: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * expansion)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
#              STREAMING ATTENTION (PATCHED)
# ============================================================

class StreamingSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        lookahead: int = 0,
        left_context: Optional[int] = None,
    ):
        super().__init__()
        self.lookahead = lookahead
        self.left_context = left_context
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        # mask cache
        self._mask_cache = {}

    def forward(self, x):
        seq_len = x.size(1)
        attn_mask = _causal_mask_cached(
            seq_len,
            device=x.device,
            lookahead=self.lookahead,
            left_context=self.left_context,
            cache=self._mask_cache,
        )
        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return out


# ============================================================
#                   CONV MODULE (UNCHANGED)
# ============================================================

class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = dropout
        self.kernel_size = kernel_size

    def _causal_depthwise_conv(self, x):
        if self.kernel_size > 1:
            x = F.pad(x, (self.kernel_size - 1, 0))
        return self.depthwise_conv(x)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)

        x = self._causal_depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(1, 2)
        return x + residual


# ============================================================
#                 FAST CONFORMER BLOCK (PATCHED)
# ============================================================

class FastConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_expansion: float = 4.0,
        conv_kernel: int = 15,
        dropout: float = 0.1,
        attn_lookahead: int = 0,
        attn_left_context: Optional[int] = None,
    ):
        super().__init__()

        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn1 = FeedForwardModule(d_model, expansion=ff_expansion, dropout=dropout)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = StreamingSelfAttention(
            d_model, nhead, dropout, attn_lookahead, attn_left_context
        )

        self.conv = ConvModule(d_model=d_model, kernel_size=conv_kernel, dropout=dropout)

        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2 = FeedForwardModule(d_model, expansion=ff_expansion, dropout=dropout)

        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x):
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))
        x = x + F.dropout(self.self_attn(self.self_attn_norm(x)), p=self.dropout, training=self.training)
        x = self.conv(x)
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        return self.final_norm(x)


# ============================================================
#            FAST CONFORMER TDT (FULLY PATCHED)
# ============================================================

class FastConformerTDT(nn.Module):
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
        use_diphone: bool = False,
        phoneme_labels: Optional[List[str]] = None,
    ):
        super().__init__()

        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.use_diphone = use_diphone
        self.phoneme_labels = phoneme_labels or list(PHONEME_LABELS)
        self.codec = DiphoneCodec(self.phoneme_labels) if self.use_diphone else None

        # optional gaussian smoothing
        self.gaussianSmoother = (
            GaussianSmoothing(neural_dim, 20, gaussianSmoothWidth, dim=1)
            if gaussianSmoothWidth > 0
            else nn.Identity()
        )
        self.inputLayerNonlinearity = nn.Softsign()

        # ===================================================
        #   DAY-SPECIFIC AFFINE LAYERS (PATCHED)
        #   Instead of ModuleList → a single Parameter tensor
        # ===================================================
        W = []
        b = []
        for _ in range(nDays):
            w = torch.eye(neural_dim)
            W.append(w)
            b.append(torch.zeros(neural_dim))

        self.day_W = nn.Parameter(torch.stack(W))      # (D, C, C)
        self.day_b = nn.Parameter(torch.stack(b))      # (D, C)

        # ===================================================
        #   SUBSAMPLING
        # ===================================================
        self.subsample = nn.Conv1d(
            neural_dim, d_model, kernel_size=kernelLen, stride=strideLen
        )
        self.subsample_norm = nn.LayerNorm(d_model)

        # ===================================================
        #   ENCODER BLOCKS
        # ===================================================
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
        out_classes = self.codec.n_diphone_classes if self.codec is not None else n_classes
        self.fc_out = nn.Linear(d_model, out_classes + 1)  # +1 for CTC blank

    # ============================================================
    #                    FORWARD PASS (PATCHED)
    # ============================================================
    def forward(self, neuralInput, dayIdx):
        # neuralInput: (B, T, C)
        x = neuralInput.transpose(1, 2)
        x = self.gaussianSmoother(x)
        x = x.transpose(1, 2)

        # ===================================================
        #     DAY-SPECIFIC LINEAR TRANSFORM (PATCHED)
        # ===================================================
        W = self.day_W[dayIdx]               # (B, C, C)
        b = self.day_b[dayIdx]               # (B, C)

        # batched matmul: (B, T, C) @ (B, C, C)^T
        x = torch.bmm(x, W.transpose(1, 2)) + b.unsqueeze(1)
        x = self.inputLayerNonlinearity(x)

        # ===================================================
        #     TEMPORAL SUBSAMPLING
        # ===================================================
        x = x.transpose(1, 2)
        x = self.subsample(x)
        x = x.transpose(1, 2)
        x = self.subsample_norm(x)

        # ===================================================
        #     ENCODER BLOCKS
        # ===================================================
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.fc_out(x)
