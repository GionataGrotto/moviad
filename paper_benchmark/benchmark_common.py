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
    "dinomaly_encoder": "deit_tiny_patch16_224.fb_in1k",
    "dinomaly_pretrained": True,
    "dinomaly_image_size": [224, 224],
    "draem_image_size": [256, 256],
    "draem_anomaly_source_path": None,
    "draem_learning_rate": 1e-4,
    "batch_size": 4,
    "num_workers": 0,
    "force_cpu_coreset": True,
    "methods": ["patchcore"],
    "epochs": 1,
    "memory_bank_size": 30000,
    "cfa_gamma_c": 4,
    "cfa_gamma_d": 1,
    "debug_max_batches": 2,
    "streaming": False,
    "padim_diag_cov": True,
    "padim_embedding_dim": 225,
    "padim_covariance_reg": 0.01,
    "padim_streaming": True,
    "clap_checkpoint": None,
    "dcase2026": {
        "score_aggregation": "temporal_topk_mean",
        "score_topk": 5,
    },
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


def percentile_decisions(
    scores: Iterable[float], percentile: float
) -> tuple[float, np.ndarray]:
    """Binarize scores using a percentile computed on the same score set.

    DCASE decisions are intentionally test-calibrated for the requested
    evaluation protocol.  The resulting threshold must therefore not be
    interpreted as an independent validation threshold.
    """
    score_array = np.asarray(list(scores), dtype=float)
    if score_array.size == 0:
        raise ValueError("Cannot calculate a percentile threshold from empty scores")
    if not np.all(np.isfinite(score_array)):
        raise ValueError(
            "Cannot calculate a percentile threshold from non-finite scores"
        )
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between 0 and 100")
    threshold = float(np.percentile(score_array, percentile))
    decisions = (score_array >= threshold).astype(int)
    return threshold, decisions


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


#: Methods whose features come from the CLAP-pretrained audio backbone.
AUDIO_BACKBONE_METHODS = frozenset({"patchcore", "padim", "cfa", "stfpm"})

#: Methods that consume the log-mel spectrogram directly, so they only need the
#: frozen frontend and never the pretrained backbone weights.
SPECTROGRAM_ONLY_METHODS = frozenset({"dinomaly", "draem"})


def check_audio_checkpoint(
    config: dict[str, Any], methods: list[str] | None = None
) -> None:
    if not config.get("pretrained", True):
        return
    if methods is not None and not AUDIO_BACKBONE_METHODS.intersection(
        {method.lower() for method in methods}
    ):
        return
    configured = config.get("clap_checkpoint")
    if configured:
        # An explicit path must be authoritative.  Silently falling back to a
        # different checkpoint makes experiments difficult to reproduce.
        candidates = [expand_path(configured)]
    else:
        if config.get("backbone", "Cnn14") == "HTSAT-base":
            candidates = [
                PROJECT_ROOT / "moviad" / "weights" / "music_speech_audioset_epoch_15_esc_89.98.pt",
                PROJECT_ROOT / "src" / "moviad" / "weights" / "music_speech_audioset_epoch_15_esc_89.98.pt",
            ]
        else:
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
    from moviad.utilities.audio.audio_feature_extractor import AudioFeatureExtractor

    return AudioFeatureExtractor(
        config["backbone"],
        config["layers"],
        device,
        frozen=frozen,
        pre_trained=config.get("pretrained", True),
        checkpoint_path=config.get("clap_checkpoint"),
    )


def make_spectrogram_transform(config: dict[str, Any]):
    """Create the frozen log-mel frontend without loading Cnn14 weights."""
    from moviad.utilities.audio.audio_feature_extractor import AudioFeatureExtractor

    _, _, spectrogram_transform = AudioFeatureExtractor._load_spectrogram_transform(
        config.get("backbone", "Cnn14")
    )
    spectrogram_transform.eval()
    return spectrogram_transform


def make_wave_to_spectrogram(config: dict[str, Any], device: torch.device, method: str):
    """Return the waveform -> log-mel transform expected by ``method``.

    Spectrogram-only methods must not instantiate the pretrained backbone: it
    would needlessly require the CLAP checkpoint just to reach its frontend.
    """
    if method.lower() in SPECTROGRAM_ONLY_METHODS:
        return make_spectrogram_transform(config)
    return make_feature_extractor(config, device, frozen=True).spectro_transform


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
            diag_cov=bool(config.get("padim_diag_cov", True)),
            img_size=input_size,
            backbone_model=make_feature_extractor(config, device, frozen=True),
            embedding_dim=int(config.get("padim_embedding_dim", 225)),
            covariance_reg=float(config.get("padim_covariance_reg", 0.01)),
        )
        return model.to(device)

    if method == "cfa":
        from moviad.models.audio.cfa.cfa import CFA

        feature_extractor = make_feature_extractor(config, device, frozen=True)
        return CFA(
            feature_extractor,
            config["backbone"],
            device,
            gamma_c=int(config.get("cfa_gamma_c", 4)),
            gamma_d=int(config.get("cfa_gamma_d", 1)),
        ).to(device)

    if method == "stfpm":
        from moviad.models.audio.stfpm.stfpm import STFPM

        teacher = make_feature_extractor(config, device, frozen=True)
        student = make_feature_extractor(config, device, frozen=False)
        return STFPM(teacher, student).to(device)

    if method == "dinomaly":
        from moviad.models.audio.dinomaly import AudioDinomaly

        image_size = tuple(
            int(value) for value in config.get("dinomaly_image_size", [224, 224])
        )
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError(
                "dinomaly_image_size must contain two positive integers."
            )
        return AudioDinomaly(
            encoder_name=config.get(
                "dinomaly_encoder", "deit_tiny_patch16_224.fb_in1k"
            ),
            device=device,
            image_size=image_size,
            pretrained=bool(config.get("dinomaly_pretrained", True)),
            spectrogram_backbone=config.get("backbone", "Cnn14"),
        ).to(device)

    if method == "draem":
        from moviad.models.audio.draem import AudioDRAEM

        image_size = tuple(
            int(value) for value in config.get("draem_image_size", [256, 256])
        )
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError("draem_image_size must contain two positive integers.")
        anomaly_source_path = config.get("draem_anomaly_source_path")
        return AudioDRAEM(
            device=device,
            image_size=image_size,
            anomaly_source_path=(
                str(expand_path(anomaly_source_path)) if anomaly_source_path else None
            ),
            spectrogram_backbone=config.get("backbone", "Cnn14"),
        ).to(device)

    raise ValueError(f"Unsupported method: {method}")


def fit_model(method: str, model, train_loader, test_loader, config: dict[str, Any], device: torch.device, debug: bool, evaluate_during_training: bool = True) -> None:
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
        trainer.train(
            train_iter,
            streaming=bool(
                config.get("padim_streaming", config.get("streaming", False))
            ),
        )
        model.eval()
        return

    if method == "cfa":
        from moviad.trainers.audio.trainer_cfa import TrainerCFA

        model.initialize_memory_bank(train_iter)
        train_iter = limited(train_loader, debug, max_batches)
        eval_iter = limited(test_loader, debug, max_batches) if evaluate_during_training else None
        trainer = TrainerCFA(model, train_iter, eval_iter, wandb=False, device=str(device))
        trainer.train(
            int(config.get("epochs", 1)),
            ["f1_img", "img_roc_auc", "pr_auc_img"] if evaluate_during_training else [],
            None,
        )
        model.eval()
        return

    if method == "stfpm":
        from moviad.trainers.audio.trainer_stfpm import TrainerSTFPM

        eval_iter = limited(test_loader, debug, max_batches) if evaluate_during_training else None
        trainer = TrainerSTFPM(model, train_iter, eval_iter, wandb=False, device=str(device))
        trainer.train(
            int(config.get("epochs", 1)),
            ["f1_img", "img_roc_auc", "pr_auc_img"] if evaluate_during_training else [],
            None,
        )
        model.eval()
        return

    if method == "dinomaly":
        from moviad.trainers.audio.trainer_dinomaly import TrainerDinomaly

        trainer = TrainerDinomaly(
            model,
            train_loader,
            device,
            debug=debug,
            max_batches=max_batches,
        )
        trainer.train(
            epochs=int(config.get("epochs", 1)),
            batch_size=int(config.get("batch_size", 1)),
        )
        model.eval()
        return

    if method == "draem":
        from moviad.trainers.audio.trainer_draem import TrainerDraem

        trainer = TrainerDraem(
            model,
            train_loader,
            device,
            debug=debug,
            max_batches=max_batches,
            learning_rate=config.get("draem_learning_rate"),
        )
        trainer.train(
            epochs=int(config.get("epochs", 1)),
            batch_size=int(config.get("batch_size", 1)),
        )
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

