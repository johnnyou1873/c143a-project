"""LightningModule wrapper for the GRU DiPhone decoder."""

from .speech_module import SpeechModule


class SpeechModuleDiPhone(SpeechModule):
    """Identical training loop to `SpeechModule`, but meant for the GRUDiphone net."""

    pass
