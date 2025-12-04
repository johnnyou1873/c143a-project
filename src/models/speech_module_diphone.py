"""LightningModule wrapper for the GRU DiPhone decoder with CTC beam search decoding."""

from typing import Dict, List, Tuple

import torch

from .speech_module import SpeechModule


class SpeechModuleDiPhone(SpeechModule):
    """Same training loop as `SpeechModule`, but uses CTC beam search for decoding."""

    def __init__(self, *args, beam_size: int = 8, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beam_size = beam_size

    def _ctc_beam_search_one_utt(
        self,
        log_probs: torch.Tensor,  # (T, C) already log-softmaxed
        input_len: int,
        beam_size: int,
        blank_idx: int = 0,
    ) -> List[List[int]]:
        """
        Simple CTC beam search that keeps a beam of token prefixes (no LM).
        Returns best sequence of token IDs (blanks removed, repeats kept).
        """
        T, C = log_probs.shape
        T = int(input_len)

        beam: Dict[Tuple[int, ...], float] = {(): 0.0}

        for t in range(T):
            frame = log_probs[t]  # (C,)
            top_k = min(beam_size, C)
            vals, idxs = frame.topk(top_k)
            vals = vals.tolist()
            idxs = idxs.tolist()

            new_beam: Dict[Tuple[int, ...], float] = {}

            for prefix, score in beam.items():
                for c, log_p in zip(idxs, vals):
                    c = int(c)
                    if c == blank_idx:
                        new_score = score + log_p
                        prev = new_beam.get(prefix, float("-inf"))
                        new_beam[prefix] = float(
                            torch.logaddexp(torch.tensor(prev), torch.tensor(new_score)).item()
                        )
                    else:
                        new_prefix = prefix + (c,)
                        new_score = score + log_p
                        prev = new_beam.get(new_prefix, float("-inf"))
                        new_beam[new_prefix] = float(
                            torch.logaddexp(torch.tensor(prev), torch.tensor(new_score)).item()
                        )

            # prune
            beam = dict(sorted(new_beam.items(), key=lambda kv: kv[1], reverse=True)[:beam_size])

        # take best prefix
        best_prefix = max(beam.items(), key=lambda kv: kv[1])[0] if beam else ()
        return [int(x) for x in best_prefix]

    def _ctc_beam_decode(self, logits: torch.Tensor, input_lengths: torch.Tensor) -> List[List[int]]:
        """Batch CTC beam search."""
        log_probs = logits.log_softmax(dim=-1)  # (B, T, C)
        batch = logits.size(0)
        decoded: List[List[int]] = []
        for i in range(batch):
            T_i = int(input_lengths[i].item())
            lp = log_probs[i, :T_i, :]
            seq = self._ctc_beam_search_one_utt(lp, input_len=T_i, beam_size=self.beam_size, blank_idx=0)
            decoded.append(seq)
        return decoded

    def validation_step(self, batch, batch_idx: int) -> None:
        neural, labels, neural_lens, label_lens, days = batch
        loss, logits, input_lengths = self._compute_loss(neural, labels, neural_lens, label_lens, days)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            pred_tokens = self._ctc_beam_decode(logits, input_lengths)
            cer_value = self._token_error_rate(pred_tokens, labels, label_lens)
            self.val_cer(cer_value)
            self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        neural, labels, neural_lens, label_lens, days = batch
        loss, logits, input_lengths = self._compute_loss(neural, labels, neural_lens, label_lens, days)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            pred_tokens = self._ctc_beam_decode(logits, input_lengths)
            cer_value = self._token_error_rate(pred_tokens, labels, label_lens)
            self.test_cer(cer_value)
            self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)
