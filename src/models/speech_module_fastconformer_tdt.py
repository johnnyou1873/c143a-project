"""LightningModule wrapper for the FastConformer-TDT encoder (streaming by default)."""

from .speech_module import SpeechModule


class SpeechModuleFastConformerTDT(SpeechModule):
    """Identical training loop to `SpeechModule`, but meant for the FastConformer-TDT net."""

    pass
