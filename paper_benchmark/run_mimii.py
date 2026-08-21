from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from moviad.datasets.subset import training_subset

from benchmark_common import (
    append_csv,
    check_audio_checkpoint,
    evaluate_model,
    expand_path,
    fit_model,
    load_config,
    make_model,
    output_dir,
    resolve_device,
    run_cli,
    set_seed,
)


MIMII_METRICS = ["img_roc_auc", "f1_img", "pr_auc_img"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style MIMII audio benchmarks.")
    parser.add_argument("--config", default="paper_benchmark/config.example.json")
    parser.add_argument("--debug", action="store_true", help="Run a tiny subset.")
    parser.add_argument("--methods", nargs="+", help="Override config methods.")
    parser.add_argument("--streaming", action="store_true", help="Fit PatchCore/PaDiM with bounded-memory streaming updates.")
    parser.add_argument(
        "--subset",
        nargs="?",
        const=0.5,
        default=1.0,
        type=float,
        metavar="FRACTION",
        help="Use a fraction of the training set (default with no value: 0.5).",
    )
    return parser.parse_args()


def machine_ids(dataset_path: Path, snr: str, category: str) -> list[str]:
    candidates = [
        dataset_path / snr / category,
        dataset_path / category,
        dataset_path,
    ]

    for base in candidates:
        if not base.exists():
            continue
        ids = sorted(
            path.name
            for path in base.iterdir()
            if path.is_dir() and path.name.startswith("id_")
        )
        if ids:
            return ids

    raise FileNotFoundError(
        f"MIMII machine directories not found under {dataset_path} for snr={snr}, category={category}"
    )


def build_loaders(
    config: dict,
    dataset_path: Path,
    snr: str,
    category: str,
    machine_id: str,
    seed: int,
    method: str,
    subset: float = 1.0,
):
    from moviad.datasets.mimi_dataset import MIMIDataset
    from moviad.utilities.configurations import Split
    from benchmark_common import make_feature_extractor

    device = resolve_device(config.get(f"{method}_device", config["device"]))
    transform = Resample(orig_freq=16000, new_freq=44100)
    feature_extractor = make_feature_extractor(config, device, frozen=True)

    train_dataset = MIMIDataset(
        dataset_path.as_posix(),
        snr,
        category,
        machine_id,
        Split.TRAIN,
        transform=transform,
        seed=seed,
    )
    train_dataset = training_subset(train_dataset, subset, seed)
    test_dataset = MIMIDataset(
        dataset_path.as_posix(),
        snr,
        category,
        machine_id,
        Split.TEST,
        wave_to_spectro=feature_extractor.spectro_transform,
        transform=transform,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )
    return train_dataset, test_dataset, train_loader, test_loader


def run_one(
    method: str,
    config: dict,
    dataset_path: Path,
    snr: str,
    category: str,
    machine_id: str,
    seed: int,
    debug: bool,
    subset: float = 1.0,
) -> dict:
    device = resolve_device(config.get(f"{method}_device", config["device"]))
    set_seed(seed)
    train_dataset, test_dataset, train_loader, test_loader = build_loaders(
        config, dataset_path, snr, category, machine_id, seed, method, subset
    )
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError(
            f"Empty split for {snr}/{category}/{machine_id}. "
            "Check the MIMII folder layout and normal/abnormal files."
        )

    input_size = tuple(int(dim) for dim in test_dataset[0][2].shape[-2:])
    model = make_model(method, config, device, input_size, class_name=category)

    started = time.perf_counter()
    fit_model(method, model, train_loader, test_loader, config, device, debug)
    metrics = evaluate_model(model, test_loader, device, MIMII_METRICS, debug=debug, max_batches=int(config.get("debug_max_batches", 2)))
    elapsed = time.perf_counter() - started

    return {
        "dataset": "mimii",
        "method": method,
        "backbone": config["backbone"],
        "layers": "|".join(config["layers"]),
        "pretrained": bool(config.get("pretrained", True)),
        "snr": snr,
        "category": category,
        "machine_id": machine_id,
        "seed": seed,
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "debug": debug,
        "elapsed_sec": round(elapsed, 3),
        "memory_bank_size": config.get("memory_bank_size"),
        "epochs": config.get("epochs"),
        **metrics,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["streaming"] = args.streaming or bool(config.get("streaming", False))
    methods = [method.lower() for method in (args.methods or config["methods"])]
    check_audio_checkpoint(config)

    mimii = config["mimii"]
    dataset_path = expand_path(mimii["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"MIMII dataset path not found: {dataset_path}")

    result_path = output_dir(config) / "mimii_results.csv"
    snrs = list(mimii["snrs"])
    categories = list(mimii["categories"])
    seeds = list(mimii["seeds"])

    if args.debug:
        snrs = snrs[:1]
        categories = categories[:1]
        seeds = seeds[:1]

    print(f"Writing MIMII results to {result_path}")
    for method in methods:
        for snr in snrs:
            for category in categories:
                ids = machine_ids(dataset_path, snr, category)
                if mimii.get("max_machines") is not None:
                    ids = ids[: int(mimii["max_machines"])]
                if args.debug:
                    ids = ids[:1]
                for seed in seeds:
                    for machine_id in ids:
                        print(f"[MIMII] method={method} snr={snr} category={category} machine={machine_id} seed={seed}")
                        row = run_one(
                            method,
                            config,
                            dataset_path,
                            snr,
                            category,
                            machine_id,
                            int(seed),
                            args.debug,
                            args.subset,
                        )
                        append_csv(result_path, row)
                        print(row)


if __name__ == "__main__":
    run_cli(main)
