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
    "force_cpu_coreset": False,
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
    "spec_band_norm": "none",
    "spec_dead_band_hz": None,
    "spec_stats_batches": 50,
    "patchcore_band_banks": False,
    "patchcore_score_std": "none",
    "padim_per_band": False,
    "padim_score_std": "none",
    "score_std_holdout": 0.1,
    "dinomaly_distance": "cosine",
    "patchcore_standardize": False,
    "patchcore_num_neighbors": 9,
    "stfpm_student_pretrained": True,
    "stfpm_optimizer": "sgd",
    "stfpm_lr": 0.4,
    "clap_checkpoint": None,
    "knn_beats_checkpoint": None,
    "knn_beats_code_dir": None,
    "knn_layers": [10],
    "knn_top_k": 1,
    "knn_metric": "cosine",
    "finetune_rank": 8,
    "finetune_epochs": 10,
    "finetune_batch_size": 8,
    "finetune_lr": 1e-4,
    "finetune_margin": 0.3,
    "handcrafted_bands": 32,
    "handcrafted_shrinkage": 0.1,
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


def prepare_audio_run(config: dict[str, Any], args, protocol: str) -> Path:
    """Resolve an optional isolated DCASE run and persist its effective settings."""
    run_name = config.get("dcase2026", {}).get("run_name")
    root = PROJECT_ROOT / "dcase2026_task2_evaluator" / protocol
    if run_name:
        if not isinstance(run_name, str) or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
            raise ValueError("dcase2026.run_name must contain only letters, digits, '_' or '-'")
        root = root / run_name
        config["output_dir"] = str(expand_path(config["output_dir"]) / protocol / run_name)
    destination = output_dir(config)
    (destination / f"{protocol}_config.json").write_text(
        json.dumps({"config": config, "arguments": vars(args)}, indent=2, default=str),
        encoding="utf-8",
    )
    return root


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

    This threshold assumes a specific anomaly prevalence (100 - percentile,
    e.g. ~1% for the default of 99): it is meant for test sets whose class
    balance matches that assumption, such as the official DCASE evaluation
    split. On a differently-balanced set (e.g. a 50/50 subset built for
    exploratory analysis) it can end up flagging far fewer or far more
    samples than the set's true anomaly rate, producing degenerate
    precision/recall/F1 regardless of how good the ranking actually is. Use
    ``optimal_f1_metrics`` instead when comparing methods' ranking quality
    independently of that calibration choice.
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


def optimal_f1_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Precision/recall/F1 at the threshold that maximises F1 on this data.

    ``percentile_decisions`` picks a threshold without looking at the labels,
    matching the DCASE evaluator's deployment-style protocol: it assumes a
    fixed expected anomaly rate and is only meaningful when the test set's
    real prevalence matches that assumption. This instead reports the best
    F1 achievable in hindsight, the same threshold-free diagnostic this
    project already uses for the MVTec benchmark (see ``cal_f1_img`` in
    ``moviad.utilities.metrics``). It ranks methods by the quality of their
    anomaly ranking rather than by an arbitrarily calibrated threshold, so it
    stays informative even on a hand-built, non-representative test subset
    (e.g. a balanced 50/50 split) where ``percentile_decisions`` degenerates
    to flagging only a handful of samples regardless of the ranking's
    quality. It picks its threshold using the very labels it is scored
    against, though, so it is optimistic and must not be reported as a
    deployment-realistic number or fed into the official DCASE evaluator.

    Returns an empty dict when ``labels`` has fewer than two classes, since
    precision/recall/F1 are undefined in that case.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if np.unique(labels).size != 2:
        return {}

    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(labels, scores)
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator,
        out=np.zeros_like(denominator), where=denominator != 0,
    )
    best = int(np.argmax(f1))
    return {
        "f1_opt": float(f1[best]),
        "precision_opt": float(precision[best]),
        "recall_opt": float(recall[best]),
    }


class LimitedBatches:
    """Re-iterable debug view: apply the limit separately for every epoch."""

    def __init__(self, iterable, max_batches):
        self.iterable = iterable
        self.max_batches = max_batches

    def __iter__(self):
        return islice(self.iterable, self.max_batches)

    def __len__(self):
        return min(len(self.iterable), self.max_batches)


def limited(iterable: Iterable[Any], debug: bool, max_batches: int) -> Iterable[Any]:
    return LimitedBatches(iterable, max_batches) if debug else iterable


def audio_sample_rate(config: dict[str, Any]) -> int:
    from moviad.utilities.audio.audio_feature_extractor import AudioFeatureExtractor

    return AudioFeatureExtractor.sample_rate_for(config.get("backbone", "Cnn14"))


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


def beats_settings(config: dict[str, Any]) -> tuple[Any, Any]:
    """(checkpoint, source folder) of BEATs; the ``knn_beats_*`` keys are accepted as a fallback."""
    return (
        config.get("beats_checkpoint") or config.get("knn_beats_checkpoint"),
        config.get("beats_code_dir") or config.get("knn_beats_code_dir"),
    )


def check_audio_checkpoint(
    config: dict[str, Any], methods: list[str] | None = None
) -> None:
    if not config.get("pretrained", True):
        return
    if config.get("backbone") == "BEATs":
        checkpoint, _ = beats_settings(config)
        if not checkpoint:
            raise ValueError("backbone 'BEATs' needs beats_checkpoint (and beats_code_dir) in the config")
        if not expand_path(checkpoint).exists():
            raise FileNotFoundError(f"BEATs checkpoint not found: {expand_path(checkpoint)}")
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


def make_feature_extractor(config: dict[str, Any], device: torch.device, frozen: bool = True, pretrained: bool | None = None):
    if config.get("backbone") == "BEATs":
        from moviad.utilities.audio.beats_feature_extractor import BEATsFeatureExtractor

        checkpoint, code_dir = beats_settings(config)
        if not checkpoint:
            raise ValueError("backbone 'BEATs' needs beats_checkpoint (and beats_code_dir) in the config")
        return BEATsFeatureExtractor(
            expand_path(checkpoint),
            code_dir=code_dir,
            layers=config["layers"],
            device=device,
            frozen=frozen,
            pretrained=config.get("pretrained", True) if pretrained is None else pretrained,
        )
    from moviad.utilities.audio.audio_feature_extractor import AudioFeatureExtractor

    return AudioFeatureExtractor(
        config["backbone"],
        config["layers"],
        device,
        frozen=frozen,
        pre_trained=config.get("pretrained", True) if pretrained is None else pretrained,
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


def method_sample_rate(method: str, config: dict[str, Any]) -> int:
    """Sample rate the waveforms must be loaded at for ``method`` (BEATs needs 16 kHz)."""
    if method.lower() in ("knn", "knn_ft", "handcrafted") or config.get("backbone") == "BEATs":
        from moviad.utilities.audio.beats_feature_extractor import BEATsFeatureExtractor

        return BEATsFeatureExtractor.sample_rate
    return audio_sample_rate(config)


def spectrogram_size(config: dict[str, Any], waveform: torch.Tensor) -> tuple[int, int]:
    """(height, width) of the time-frequency map the anomaly maps are reported at."""
    if config.get("backbone") == "BEATs":
        from moviad.utilities.audio.beats_feature_extractor import BEATsFeatureExtractor

        return BEATsFeatureExtractor.fbank_shape(waveform.shape[-1])
    spectrogram = make_spectrogram_transform(config)(waveform.unsqueeze(0))
    return tuple(int(dim) for dim in spectrogram.shape[-2:])


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
            num_neighbors=int(config.get("patchcore_num_neighbors", 9)),
            memory_bank_size=int(config["memory_bank_size"]),
            blur=True,
            standardize_features=bool(config.get("patchcore_standardize", False)),
            band_banks=bool(config.get("patchcore_band_banks", False)),
            score_standardization=config.get("patchcore_score_std", "none"),
            holdout_fraction=float(config.get("score_std_holdout", 0.1)),
        ).to(device)

    if method == "knn_ft":
        from moviad.models.audio.knn import BEATsEncoder
        from moviad.models.audio.knn.finetune import FineTunedKNN

        if not config.get("knn_beats_checkpoint"):
            raise ValueError("Method 'knn_ft' needs knn_beats_checkpoint (and knn_beats_code_dir) in the config")
        encoder = BEATsEncoder(
            expand_path(config["knn_beats_checkpoint"]),
            code_dir=config.get("knn_beats_code_dir"),
            layers=config.get("knn_layers", [10]),
            device=device,
        )
        return FineTunedKNN(
            encoder, device,
            top_k=int(config.get("knn_top_k", 1)), metric=config.get("knn_metric", "cosine"),
            rank=int(config.get("finetune_rank", 8)), targets=tuple(config.get("finetune_targets", ["q_proj", "v_proj"])),
            epochs=int(config.get("finetune_epochs", 10)), batch_size=int(config.get("finetune_batch_size", 8)), lr=float(config.get("finetune_lr", 1e-4)),
            margin=float(config.get("finetune_margin", 0.3)), seed=int(config.get("seed", 0)),
        )

    if method == "handcrafted":
        from moviad.models.audio.knn import SubbandMahalanobis, SubbandStatsEncoder

        encoder = SubbandStatsEncoder(int(config.get("handcrafted_bands", 32)), device=device)
        return SubbandMahalanobis(encoder, device, shrinkage=float(config.get("handcrafted_shrinkage", 0.1)))

    if method == "knn":
        from moviad.models.audio.knn import AudioKNN, BEATsEncoder

        if not config.get("knn_beats_checkpoint"):
            raise ValueError("Method 'knn' needs knn_beats_checkpoint (and knn_beats_code_dir) in the config")
        encoder = BEATsEncoder(
            expand_path(config["knn_beats_checkpoint"]),
            code_dir=config.get("knn_beats_code_dir"),
            layers=config.get("knn_layers", [10]),
            device=device,
        )
        return AudioKNN(
            encoder,
            device,
            top_k=int(config.get("knn_top_k", 1)),
            metric=config.get("knn_metric", "cosine"),
        )

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
            embedding_dim=min(
                int(config.get("padim_embedding_dim", 225)),
                Padim.lookup_embedding_size(config["backbone"], config["layers"])[0],
            ),
            covariance_reg=float(config.get("padim_covariance_reg", 0.01)),
            per_band=bool(config.get("padim_per_band", False)),
            score_standardization=config.get("padim_score_std", "none"),
            holdout_fraction=float(config.get("score_std_holdout", 0.1)),
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
        # The STFPM paper trains a randomly initialised student against the pretrained teacher;
        # starting from the same weights makes the student already match the teacher everywhere.
        student = make_feature_extractor(
            config, device, frozen=False, pretrained=bool(config.get("stfpm_student_pretrained", True))
        )
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
            distance=config.get("dinomaly_distance", "cosine"),
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


def _spectrogram_consumers(method: str, model) -> list:
    """Objects of ``model`` that turn waveforms into log-mel spectrograms."""
    method = method.lower()
    if method in ("patchcore", "cfa"):
        return [model.feature_extractor]
    if method == "padim":
        return [model.backbone_model]
    if method == "stfpm":
        return [model.teacher, model.student]
    if method == "dinomaly":
        return [model]
    raise ValueError(f"Spectrogram preprocessing is not supported for method {method!r}")


def attach_spectrogram_preprocessing(method: str, model, train_loader, config: dict[str, Any], device: torch.device) -> None:
    """Fit and attach the optional band standardisation / dead-band filling (off by default)."""
    band_norm = config.get("spec_band_norm", "none")
    dead_hz = config.get("spec_dead_band_hz")
    if band_norm == "none" and dead_hz is None:
        return
    if config.get("backbone") == "BEATs":
        raise ValueError("spec_band_norm / spec_dead_band_hz apply to the Cnn14 log-mel frontend, not to BEATs")
    from itertools import islice

    from moviad.models.audio.audio_vad_model import AudioVADModel
    from moviad.models.audio.components.spectrogram_preprocess import SpectrogramPreprocessor, first_band_above

    n_mels, fmin, fmax = 64, 50, 14000  # the log-mel frontend of the Cnn14 / HTSAT backbones
    dead_start = None if dead_hz is None else first_band_above(float(dead_hz), n_mels, fmin, fmax)
    preprocessor = SpectrogramPreprocessor(n_mels, band_norm, dead_start).to(device)
    if band_norm != "none":
        transform = make_spectrogram_transform(config).to(device)
        batches = islice(train_loader, int(config.get("spec_stats_batches", 50)))
        with torch.no_grad():
            preprocessor.fit(transform(AudioVADModel.batch_input(batch).to(device)) for batch in batches)
    print(f"Spectrogram preprocessing: band_norm={band_norm}, dead bands from index {dead_start}", flush=True)
    for consumer in _spectrogram_consumers(method, model):
        consumer.set_spectrogram_preprocessor(preprocessor)


def fit_model(method: str, model, train_loader, test_loader, config: dict[str, Any], device: torch.device, debug: bool, evaluate_during_training: bool = True) -> None:
    method = method.lower()
    max_batches = int(config.get("debug_max_batches", 2))
    train_iter = limited(train_loader, debug, max_batches)
    attach_spectrogram_preprocessing(method, model, train_loader, config, device)

    if method == "patchcore":
        from moviad.trainers.audio.trainer_patchcore import TrainerPatchCore

        model.train()
        trainer = TrainerPatchCore(
            model,
            train_iter,
            test_loader,
            device,
            force_cpu=bool(config.get("force_cpu_coreset", False)),
        )
        trainer.train(streaming=bool(config.get("streaming", False)))
        model.eval()
        return

    if method == "knn_ft":
        model.fit(train_loader)  # needs the dataset (attributes), not a plain batch iterator
        model.eval()
        return

    if method in ("knn", "handcrafted"):
        model.fit(train_iter)
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
        trainer = TrainerSTFPM(
            model, train_iter, eval_iter, wandb=False, device=str(device),
            optimizer=config.get("stfpm_optimizer", "sgd"), learning_rate=float(config.get("stfpm_lr", 0.4)),
        )
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

