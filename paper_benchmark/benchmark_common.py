from __future__ import annotations

import csv
import json
import os
import random
import sys
import traceback
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if import_root.exists() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


DEFAULT_CONFIG: dict[str, Any] = {
    "device": "auto",
    "output_dir": "results/paper_benchmark",
    "backbone": "Cnn14",
    "layers": ["conv_block2", "conv_block3", "conv_block4"],
    "pretrained": True,
    "batch_size": 4,
    "num_workers": 0,
    "force_cpu_coreset": True,
    "methods": ["patchcore"],
    "epochs": 1,
    "memory_bank_size": 30000,
    "debug_max_batches": 2,
    "streaming": False,
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path:
        with Path(path).open("r", encoding="utf-8") as fh:
            config = deep_update(config, json.load(fh))
    return config


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def output_dir(config: dict[str, Any]) -> Path:
    path = expand_path(config["output_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited(iterable: Iterable[Any], debug: bool, max_batches: int) -> Iterable[Any]:
    return islice(iterable, max_batches) if debug else iterable


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames = list(row.keys())
    if not write_header:
        # Keep one stable schema across methods. Different models expose
        # different optional metrics, so using the current row's key order can
        # shift values into the wrong columns when appending.
        with path.open("r", newline="", encoding="utf-8") as fh:
            fieldnames = next(csv.reader(fh), fieldnames)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def check_audio_checkpoint(config: dict[str, Any]) -> None:
    if not config.get("pretrained", True):
        return
    candidates = [
        PROJECT_ROOT / "moviad" / "weights" / "clap_encoder.pth",
        PROJECT_ROOT / "src" / "moviad" / "weights" / "clap_encoder.pth",
    ]
    if not any(path.exists() for path in candidates):
        expected = " or ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Missing CLAP checkpoint. Expected: "
            f"{expected}. Set pretrained=false only for pipeline debugging."
        )


def make_feature_extractor(config: dict[str, Any], device: torch.device, frozen: bool = True):
    from moviad.utilities.audio.audio_feature_exctractor import AudioFeatureExtractor

    return AudioFeatureExtractor(
        config["backbone"],
        config["layers"],
        device,
        frozen=frozen,
        pre_trained=config.get("pretrained", True),
    )


def make_model(
    method: str,
    config: dict[str, Any],
    device: torch.device,
    input_size: tuple[int, int],
    class_name: str,
):
    method = method.lower()
    if method == "patchcore":
        from moviad.models.audio.patchcore.patchcore import PatchCore

        feature_extractor = make_feature_extractor(config, device, frozen=True)
        return PatchCore(
            device=device,
            input_size=input_size,
            feature_extractor=feature_extractor,
            memory_bank_size=int(config["memory_bank_size"]),
            blur=True,
        ).to(device)

    if method == "padim":
        from moviad.models.audio.padim.padim import Padim

        model = Padim(
            backbone_model_name=config["backbone"],
            class_name=class_name,
            device=device,
            layers_idxs=config["layers"],
            diag_cov=False,
            img_size=input_size,
            backbone_model=make_feature_extractor(config, device, frozen=True),
        )
        return model.to(device)

    if method == "cfa":
        from moviad.models.audio.cfa.cfa import CFA

        feature_extractor = make_feature_extractor(config, device, frozen=True)
        return CFA(feature_extractor, config["backbone"], device).to(device)

    if method == "stfpm":
        from moviad.models.audio.stfpm.stfpm import STFPM

        teacher = make_feature_extractor(config, device, frozen=True)
        student = make_feature_extractor(config, device, frozen=False)
        return STFPM(teacher, student).to(device)

    raise ValueError(f"Unsupported method: {method}")


def fit_model(method: str, model, train_loader, test_loader, config: dict[str, Any], device: torch.device, debug: bool) -> None:
    method = method.lower()
    max_batches = int(config.get("debug_max_batches", 2))
    train_iter = limited(train_loader, debug, max_batches)

    if method == "patchcore":
        from moviad.trainers.audio.trainer_patchcore import TrainerPatchCore

        model.train()
        trainer = TrainerPatchCore(
            model,
            train_iter,
            test_loader,
            device,
            force_cpu=bool(config.get("force_cpu_coreset", True)),
        )
        trainer.train(streaming=bool(config.get("streaming", False)))
        model.eval()
        return

    if method == "padim":
        from moviad.trainers.audio.trainer_padim import PadimTrainer

        trainer = PadimTrainer(model=model, device=device, save_path=None, data_path=None, class_name="benchmark")
        trainer.train(train_iter, streaming=bool(config.get("streaming", False)))
        model.eval()
        return

    if method == "cfa":
        from moviad.trainers.audio.trainer_cfa import TrainerCFA

        model.initialize_memory_bank(train_iter)
        train_iter = limited(train_loader, debug, max_batches)
        eval_iter = limited(test_loader, debug, max_batches)
        trainer = TrainerCFA(model, train_iter, eval_iter, wandb=False, device=str(device))
        trainer.train(int(config.get("epochs", 1)), ["f1_img", "img_roc_auc", "pr_auc_img"], None)
        model.eval()
        return

    if method == "stfpm":
        from moviad.trainers.audio.trainer_stfpm import TrainerSTFPM

        eval_iter = limited(test_loader, debug, max_batches)
        trainer = TrainerSTFPM(model, train_iter, eval_iter, wandb=False, device=str(device))
        trainer.train(int(config.get("epochs", 1)), ["f1_img", "img_roc_auc", "pr_auc_img"], None)
        model.eval()
        return

    raise ValueError(f"Unsupported method: {method}")


def evaluate_model(
    model,
    test_loader,
    device: torch.device,
    metrics: list[str],
    debug: bool = False,
    max_batches: int = 2,
    binarize_masks=None,
) -> dict[str, float]:
    from moviad.utilities.evaluator import Evaluator

    test_iter = limited(test_loader, debug, max_batches)
    evaluator = Evaluator(test_iter, device)
    evaluate_kwargs = {
        'metrics_to_compute': metrics,
        'metrics_to_dict': True,
    }
    if binarize_masks is not None:
        evaluate_kwargs['binarize_masks'] = binarize_masks
    result = evaluator.evaluate(model, **evaluate_kwargs)
    return {key: float(value) for key, value in result.items()}


def run_cli(main_func) -> None:
    exit_code = 0
    try:
        main_func()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

