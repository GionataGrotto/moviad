"""Does a trivial signal statistic already separate normal from anomalous clips?

For every machine, the source-domain test clips are described by a few global statistics (RMS
energy, spectral centroid, spectral flatness, high-frequency energy share) and the AUC of each
statistic is computed against the anomaly label. An AUC far from 0.5 means the statistic alone
carries label information; an AUC clearly below 0.5 means anomalous clips get *lower* values.
This explains systematic effects such as detectors that score below chance on one machine, and
tells whether a per-clip normalisation of the input is worth trying.

    python paper_benchmark/diagnose_signal_features.py --dataset-path ~/datasets/DCASE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from moviad.datasets.dcase2026_task2 import DCASE2026Task2Dataset, discover_machine_types  # noqa: E402

FEATURES = ("rms", "spectral_centroid", "spectral_flatness", "high_freq_share")


def clip_statistics(waveform: torch.Tensor, sample_rate: int = 16000) -> dict[str, float]:
    spectrum = torch.stft(waveform, n_fft=1024, hop_length=512, window=torch.hann_window(1024), return_complex=True).abs()
    power = spectrum**2 + 1e-12  # (frequency, time)
    frequencies = torch.linspace(0, sample_rate / 2, power.shape[0]).unsqueeze(1)
    mean_power = power.mean(dim=1)
    return {
        "rms": float(waveform.pow(2).mean().sqrt()),
        "spectral_centroid": float(((frequencies * power).sum(dim=0) / power.sum(dim=0)).mean()),
        "spectral_flatness": float((torch.exp(torch.log(power).mean(dim=0)) / power.mean(dim=0)).mean()),
        "high_freq_share": float(mean_power[frequencies[:, 0] > 4000].sum() / mean_power.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--channel", default="0", help="0, 1 or 'diff'")
    parser.add_argument("--domain", choices=("source", "target", "all"), default="source")
    args = parser.parse_args()
    channel = args.channel if args.channel == "diff" else int(args.channel)

    print(f"{'machine':12s} " + " ".join(f"{name:>18s}" for name in FEATURES))
    for machine in discover_machine_types(args.dataset_path):
        dataset = DCASE2026Task2Dataset(
            args.dataset_path, machine, "test", domain=args.domain, channel=channel, target_sample_rate=16000
        )
        labels, values = [], {name: [] for name in FEATURES}
        for waveform, label, _ in dataset:
            labels.append(label)
            for name, value in clip_statistics(waveform).items():
                values[name].append(value)
        print(f"{machine:12s} " + " ".join(f"{roc_auc_score(labels, values[name]):18.3f}" for name in FEATURES))


if __name__ == "__main__":
    main()
