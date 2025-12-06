"""
Run an evaluation pass of the frozen diphone GRU + phoneme seq2seq LM and report CER.

Defaults match the Hydra config in `configs/model/speech_with_ps2s.yaml`, but can be
overridden via CLI flags.
"""

import argparse
import sys
from pathlib import Path

import torch
from edit_distance import SequenceMatcher

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.speech_datamodule import SpeechDataModule  # noqa: E402
from src.models.components.gru_diphone import GRUDiphone  # noqa: E402
from src.models.speech_module_with_ps2s import SpeechModuleCTCWithPS2SLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate diphone GRU + phoneme seq2seq LM (CER only).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "allDatasets.pkl",
        help="Pickled dataset path (matches configs/data/speech.yaml).",
    )
    parser.add_argument(
        "--gru-ckpt",
        type=Path,
        default=PROJECT_ROOT / "data" / "gru1024-diphone.ckpt",
        help="Frozen diphone GRU checkpoint.",
    )
    parser.add_argument(
        "--ps2s-ckpt",
        type=Path,
        default=PROJECT_ROOT / "data" / "phoneme_seq2seq.pt",
        help="Pretrained phoneme seq2seq checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Eval batch size.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap on number of batches.")
    parser.add_argument("--device", type=str, default="auto", help="Force device (cpu|cuda); auto picks if available.")
    return parser.parse_args()


def build_model(gru_ckpt: Path, ps2s_ckpt: Path, n_input_features: int, n_classes: int, n_days: int) -> SpeechModuleCTCWithPS2SLM:
    """Instantiate the LightningModule with frozen GRU + loaded seq2seq LM."""
    net = GRUDiphone(
        neural_dim=n_input_features,
        n_classes=n_classes,
        hidden_dim=1024,
        layer_dim=5,
        nDays=n_days,
        dropout=0.4,
        strideLen=4,
        kernelLen=32,
        gaussianSmoothWidth=2.0,
        bidirectional=False,
    )

    # Optimizer/scheduler placeholders: not used for evaluation
    dummy_opt = lambda params: torch.optim.Adam(params, lr=1e-4)

    model = SpeechModuleCTCWithPS2SLM(
        net=net,
        optimizer=dummy_opt,
        scheduler=None,
        gru_ckpt_path=str(gru_ckpt),
        ps2s_model_path=str(ps2s_ckpt),
        ps2s_max_len=128,
        attention_mode="bidir",
        lookahead=1,
    )
    return model


def strip_special_tokens(module: SpeechModuleCTCWithPS2SLM, token_row: torch.Tensor) -> list[int]:
    seq: list[int] = []
    for t in token_row.tolist():
        if t in (module.pad_idx, module.sos_idx, module.eos_idx):
            if t == module.eos_idx:
                break
            continue
        seq.append(int(t))
    return seq


@torch.no_grad()
def evaluate(model: SpeechModuleCTCWithPS2SLM, datamodule: SpeechDataModule, device: torch.device, max_batches: int | None) -> float:
    """Loop over the test set and compute global CER."""
    loader = datamodule.test_dataloader()
    model.eval()
    model.to(device)

    total_dist = 0
    total_len = 0
    example_logged = False

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        neural, labels, neural_lens, label_lens, days = batch
        neural = neural.to(device)
        labels = labels.to(device)
        neural_lens = neural_lens.to(device)
        label_lens = label_lens.to(device)
        days = days.to(device)

        # 1) GRU forward -> CTC decode (no grad)
        ctc_logits = model.net(neural, days)
        out_lengths = model._compute_output_lengths(neural_lens)
        noisy_ctc_ids = model._ctc_decode(ctc_logits, out_lengths)

        # 2) Build clean LM targets (from ground-truth CTC labels)
        target_lm_ids: list[list[int]] = []
        for seq, L in zip(labels, label_lens):
            seq_ids = [int(t) for t in seq[: int(L)] if int(t) > 0]
            target_lm_ids.append(model._ids_to_seq2seq_tokens(seq_ids))

        # 3) Build LM src batch + autoregressive decode
        src_batch = model._build_src_batch(noisy_ctc_ids, device=device)
        model.seq2seq.eval()
        _, pred_token_seqs = model._autoregressive_forward(src_batch, return_tokens=True)
        if pred_token_seqs is None:
            continue

        # 4) Accumulate CER
        for pred_row, tgt_ids in zip(pred_token_seqs, target_lm_ids):
            pred_ids = strip_special_tokens(model, pred_row)
            if len(tgt_ids) == 0:
                continue
            dist = SequenceMatcher(a=tgt_ids, b=pred_ids).distance()
            total_dist += dist
            total_len += len(tgt_ids)

        # Log one example for sanity
        if not example_logged and len(src_batch) > 0 and len(target_lm_ids) > 0:
            noisy_txt = " ".join(model._map_ids_to_symbols(src_batch[0]))
            tgt_txt = " ".join(model.idx_to_phoneme.get(t, f"<{t}>") for t in target_lm_ids[0])
            pred_txt = " ".join(
                model.idx_to_phoneme.get(t, f"<{t}>") for t in strip_special_tokens(model, pred_token_seqs[0])
            )
            print(f"[example] noisy: {noisy_txt}")
            print(f"[example] pred : {pred_txt}")
            print(f"[example] tgt  : {tgt_txt}")
            example_logged = True

    if total_len == 0:
        return 0.0
    return total_dist / total_len


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device if args.device != "auto" else "cpu")

    datamodule = SpeechDataModule(
        dataset_path=str(args.dataset),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
        shuffle=False,
        persistent_workers=False,
        white_noise_std=0.0,
        constant_offset_std=0.0,
        channel_dropout_p=0.0,
    )
    datamodule.setup(stage="test")

    model = build_model(
        gru_ckpt=args.gru_ckpt,
        ps2s_ckpt=args.ps2s_ckpt,
        n_input_features=datamodule.n_input_features,
        n_classes=datamodule.n_classes,
        n_days=datamodule.n_days,
    )

    cer = evaluate(model, datamodule, device=device, max_batches=args.max_batches)
    print(f"\nFinal CER: {cer:.4f}")


if __name__ == "__main__":
    main()
