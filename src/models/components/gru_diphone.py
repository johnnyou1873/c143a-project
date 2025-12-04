from typing import List, Optional

import torch
from torch import nn

from ...utils.augmentations import GaussianSmoothing

# ============================================================
#               PHONEME LABELS (EXACTLY AS GIVEN)
# ============================================================

PHONEME_LABELS: List[str] = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "<sil>",
]


# ============================================================
#                 DIPHONE CODEC (ID-BASED)
# ============================================================

class DiphoneCodec:
    """Utility for converting between monophone ID sequences and diphone ID sequences."""

    def __init__(self, phoneme_labels: List[str], sil_token: str = "<sil>") -> None:
        self.phoneme_labels = phoneme_labels
        self.n_phones = len(phoneme_labels)
        self.sil_token = sil_token
        if sil_token not in phoneme_labels:
            raise ValueError(f"sil_token {sil_token} not in phoneme_labels")
        self.sil_id = phoneme_labels.index(sil_token)

    @property
    def n_diphone_classes(self) -> int:
        return self.n_phones * self.n_phones

    def mono_ids_to_diphone_ids(
        self,
        mono_ids: List[int],
        include_sil_diphones: bool = True,
    ) -> List[int]:
        out: List[int] = []
        if len(mono_ids) < 2:
            return out

        for i in range(len(mono_ids) - 1):
            a = mono_ids[i]
            b = mono_ids[i + 1]

            if not include_sil_diphones and (a == self.sil_id or b == self.sil_id):
                continue

            d_id = a * self.n_phones + b
            out.append(d_id)

        return out

    def diphone_ids_to_mono_ids(self, diphone_ids: List[int]) -> List[int]:
        if not diphone_ids:
            return []

        monos: List[int] = []
        first_d = diphone_ids[0]
        first_a = first_d // self.n_phones
        monos.append(first_a)

        for d_id in diphone_ids:
            b = d_id % self.n_phones
            monos.append(b)

        return monos

    def clean_mono_ids(
        self,
        mono_ids: List[int],
        remove_sil: bool = True,
        collapse_repeats: bool = True,
    ) -> List[int]:
        out: List[int] = []
        prev: Optional[int] = None

        for i in mono_ids:
            if remove_sil and i == self.sil_id:
                continue
            if collapse_repeats and i == prev:
                continue
            out.append(i)
            prev = i

        return out

    def ids_to_labels(self, mono_ids: List[int]) -> List[str]:
        return [self.phoneme_labels[i] for i in mono_ids]


class GRUDiphone(nn.Module):
    """GRU encoder with day-specific input affine layers (for diphone modeling)."""

    def __init__(
        self,
        neural_dim: int,
        n_classes: int,
        hidden_dim: int,
        layer_dim: int,
        nDays: int = 24,
        dropout: float = 0.0,
        device: str = "cuda",
        strideLen: int = 4,
        kernelLen: int = 14,
        gaussianSmoothWidth: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        # model hyperparameters
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.codec = DiphoneCodec(PHONEME_LABELS, sil_token="<sil>")
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional

        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )

        if self.gaussianSmoothWidth > 0:
            self.gaussianSmoother = GaussianSmoothing(
                neural_dim, 20, self.gaussianSmoothWidth, dim=1
            )
        else:
            self.gaussianSmoother = torch.nn.Identity()

        # day-specific affine transform
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        for x in range(nDays):
            self.dayWeights.data[x].copy_(torch.eye(neural_dim))

        # Layer normalization
        norm_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.layerNorm = nn.LayerNorm(norm_dim)

        # GRU layers
        self.gru_decoder = nn.GRU(
            neural_dim * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # output projection (+1 for CTC blank)
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(hidden_dim * 2, n_classes + 1)
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)

    def forward(self, neuralInput: torch.Tensor, dayIdx: torch.Tensor) -> torch.Tensor:
        device = neuralInput.device
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)),
            (0, 2, 1),
        )

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # normalize
        hid = self.layerNorm(hid)

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out


# aliases for backward compatibility with earlier naming
GRUDecoderDiPhone = GRUDiphone
GRUDecoderDiPH = GRUDiphone


# ------------------------------------------------------------
# Module-level helpers (using the default codec definition)
# ------------------------------------------------------------
_default_codec = DiphoneCodec(PHONEME_LABELS, sil_token="<sil>")


def monophones_to_diphone_ids(
    mono_ids: List[int],
    include_sil_diphones: bool = True,
) -> List[int]:
    """Convert monophone IDs to diphone IDs with a shared default codec."""
    return _default_codec.mono_ids_to_diphone_ids(mono_ids, include_sil_diphones)


def diphones_to_monophones(
    diphone_ids: List[int],
    remove_sil: bool = True,
    collapse_repeats: bool = True,
) -> List[int]:
    """
    Convert diphone IDs back to monophone IDs and optionally:
    - remove <sil>
    - collapse repeats (CTC-style)
    """
    mono_ids = _default_codec.diphone_ids_to_mono_ids(diphone_ids)
    return _default_codec.clean_mono_ids(
        mono_ids, remove_sil=remove_sil, collapse_repeats=collapse_repeats
    )


def monophone_ids_to_labels(mono_ids: List[int]) -> List[str]:
    """Convert monophone ID list to string labels using the default codec."""
    return _default_codec.ids_to_labels(mono_ids)
