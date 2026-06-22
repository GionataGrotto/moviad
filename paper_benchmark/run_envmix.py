from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from benchmark_common import (
    append_csv,
    check_audio_checkpoint,
    evaluate_model,
    expand_path,
    fit_model,
    load_config,
    make_feature_extractor,
    make_model,
    output_dir,
    resolve_device,
    run_cli,
    set_seed,
)


ENVMIX_METRICS = [
    "img_roc_auc",
    "f1_img",
    "pr_auc_img",
    "per_pixel_rocauc",
    "f1_pxl",
    "pr_auc_pxl",
    "au_pro_pxl",
    "mse_pxl",
    "f1_tmp",
    "tmp_auc_roc",
    "pr_auc_tmp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style EnvMix audio benchmarks.")
    parser.add_argument("--config", default="paper_benchmark/config.example.json")
    parser.add_argument("--debug", action="store_true", help="Run a tiny subset.")
    parser.add_argument("--methods", nargs="+", help="Override config methods.")
    return parser.parse_args()


def build_loaders(config: dict, background_category: str, snr_db: float, seed: int, debug: bool):
    from moviad.datasets.audio_dataset import generate_urban_esc_V1

    device = resolve_device(config["device"])
    envmix = config["envmix"]
    feature_extractor = make_feature_extractor(config, device, frozen=True)
    max_samples = envmix.get("max_samples_debug") if debug else None

    train_dataset, test_dataset, _ = generate_urban_esc_V1(
        urban_category=background_category,
        esc50_categories=envmix["esc50_anomaly_categories"],
        wav_to_spectro=feature_extractor.spectro_transform,
        SNR_dB=float(snr_db),
        path_urban=expand_path(envmix["urban_path"]),
        path_esc50=expand_path(envmix["esc50_path"]),
        seed=int(seed),
        max_num_samples=max_samples,
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


def run_one(method: str, config: dict, background_category: str, snr_db: float, seed: int, debug: bool) -> dict:
    device = resolve_device(config["device"])
    set_seed(seed)
    train_dataset, test_dataset, train_loader, test_loader = build_loaders(
        config, background_category, snr_db, seed, debug
    )
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError(f"Empty EnvMix split for background={background_category}, seed={seed}.")

    input_size = tuple(int(dim) for dim in test_dataset.img_shape[1:])
    model = make_model(method, config, device, input_size, class_name=background_category)

    started = time.perf_counter()
    fit_model(method, model, train_loader, test_loader, config, device, debug)
    metrics = evaluate_model(model, test_loader, device, ENVMIX_METRICS)
    elapsed = time.perf_counter() - started

    return {
        "dataset": "envmix",
        "method": method,
        "backbone": config["backbone"],
        "layers": "|".join(config["layers"]),
        "pretrained": bool(config.get("pretrained", True)),
        "background_category": background_category,
        "snr_db": snr_db,
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
    methods = [method.lower() for method in (args.methods or config["methods"])]
    check_audio_checkpoint(config)

    envmix = config["envmix"]
    urban_path = expand_path(envmix["urban_path"])
    esc50_path = expand_path(envmix["esc50_path"])
    if not esc50_path.exists():
        raise FileNotFoundError(f"ESC-50 path not found: {esc50_path}")
    if not urban_path.exists():
        print(f"UrbanSound8K path not found: {urban_path}. The dataset loader may try to download it.")

    backgrounds = list(envmix["background_categories"])
    snrs = list(envmix["snr_db"])
    seeds = list(envmix["seeds"])
    if args.debug:
        backgrounds = backgrounds[:1]
        snrs = snrs[:1]
        seeds = seeds[:1]

    result_path = output_dir(config) / "envmix_results.csv"
    print(f"Writing EnvMix results to {result_path}")
    for method in methods:
        for background_category in backgrounds:
            for snr_db in snrs:
                for seed in seeds:
                    print(f"[EnvMix] method={method} background={background_category} snr_db={snr_db} seed={seed}")
                    row = run_one(method, config, background_category, float(snr_db), int(seed), args.debug)
                    append_csv(result_path, row)
                    print(row)


if __name__ == "__main__":
    run_cli(main)
