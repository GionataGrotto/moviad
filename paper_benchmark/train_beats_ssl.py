"""Train one self-supervised BEATs adapter on the normal training clips of several machines and save it.

    uv run python paper_benchmark/train_beats_ssl.py --config paper_benchmark/config.knn.json --out adapters/ssl_reg.pt \
        --objective reg --signal ls --epochs 5

The adapter is then used by every detector through ``"beats_adapter": "adapters/ssl_reg.pt"`` in the config. Adapting per
machine instead needs no script: set ``"beats_ssl": {"objective": "reg", "signal": "ls", "epochs": 5}``. The training
clips of the dev machines and of the evaluation machines are different sets: train one adapter for each
(``--dataset-path``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from benchmark_common import expand_path, load_config, make_feature_extractor, resolve_device
from moviad.datasets.dcase2026_task2 import DCASE2026Task2Dataset, discover_machine_types
from moviad.models.audio.beats_ssl import OBJECTIVES, SIGNALS, adapt_beats, save_adapter
from moviad.utilities.audio.beats_feature_extractor import BEATsFeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-path", default=None, help="default: dcase2026.dataset_path of the config")
    parser.add_argument("--machines", nargs="+", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objective", choices=OBJECTIVES, default="reg")
    parser.add_argument("--signal", choices=SIGNALS, default="near")
    parser.add_argument("--train-domains", choices=("source", "target", "all"), default="source")
    parser.add_argument("--layers", nargs="+", type=int, default=[8, 10, 12])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    config.update({"backbone": "BEATs", "layers": args.layers, "beats_adapter": None})
    dataset_path = expand_path(args.dataset_path or config["dcase2026"]["dataset_path"])
    machines = args.machines or discover_machine_types(dataset_path)
    sets = [
        DCASE2026Task2Dataset(dataset_path, m, "train", train_domains=args.train_domains, channel="stereo", target_sample_rate=BEATsFeatureExtractor.sample_rate)
        for m in machines
    ]
    print(f"Adapting BEATs on {sum(len(s) for s in sets)} clips of {len(machines)} machines: {machines}", flush=True)
    device = resolve_device(config["device"])
    extractor = make_feature_extractor(config, device, frozen=True)
    history = adapt_beats(
        extractor, ConcatDataset(sets), objective=args.objective, signal=args.signal, epochs=args.epochs, lr=args.lr, rank=args.rank,
        alpha=args.alpha, batch_size=args.batch_size, mask_ratio=args.mask_ratio, temperature=args.temperature, seed=args.seed,
    )
    save_adapter(extractor, args.out, rank=args.rank, alpha=args.alpha, objective=args.objective, signal=args.signal, machines=machines,
                 loss=history, layers=args.layers)
    print(f"Adapter saved to {args.out}")


if __name__ == "__main__":
    main()
