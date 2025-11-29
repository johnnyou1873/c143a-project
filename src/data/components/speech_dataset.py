from typing import Callable, Iterable, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    """Dataset wrapper for neural speech features and phoneme labels."""

    def __init__(self, data: Iterable, transform: Optional[Callable] = None) -> None:
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum(len(day["sentenceDat"]) for day in data)

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.phone_seqs.append(data[day]["phonemes"][trial])
                self.neural_time_bins.append(int(data[day]["sentenceDat"][trial].shape[0]))
                self.phone_seq_lens.append(int(data[day]["phoneLens"][trial]))
                self.days.append(day)

    def __len__(self) -> int:
        return self.n_trials

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int64),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int64),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int64),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )


def speech_collate_fn(batch):
    """Pad variable length sequences for CTC training."""
    neural, labels, neural_lens, label_lens, days = zip(*batch)
    neural_padded = pad_sequence(neural, batch_first=True, padding_value=0.0)
    label_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    return (
        neural_padded,
        label_padded,
        torch.stack(neural_lens),
        torch.stack(label_lens),
        torch.stack(days),
    )
