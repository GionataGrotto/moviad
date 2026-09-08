"""DCASE 2026 Task 2 far-to-near self-supervised benchmark.

This runner is intentionally separate from ``run_dcase2026_task2.py``.  It
pretrains an encoder on normal source recordings from the far microphone and
then trains the selected anomaly detector on normal source recordings from the
near microphone.  Test outputs are written to a separate evaluator tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader

try:
    from .benchmark_common import (
        check_audio_checkpoint,
        expand_path,
        fit_model,
        load_config,
        make_feature_extractor,
        make_model,
        make_spectrogram_transform,
        output_dir,
        percentile_decisions,
        resolve_device,
        run_cli,
        set_seed,
    )
    from .run_dcase2026_task2 import (
        _score_model,
        _write_pairs,
    )
except ImportError:  # supports ``python paper_benchmark/run_*.py``
    from benchmark_common import (
        check_audio_checkpoint,
        expand_path,
        fit_model,
        load_config,
        make_feature_extractor,
        make_model,
        make_spectrogram_transform,
        output_dir,
        percentile_decisions,
        resolve_device,
        run_cli,
        set_seed,
    )
    from run_dcase2026_task2 import (
        _score_model,
        _write_pairs,
    )

from moviad.datasets.dcase2026_task2 import (
    DCASE2026Task2Dataset,
    discover_machine_types,
)


AUDIO_SSL_METHODS = frozenset({"padim", "patchcore", "stfpm", "cfa"})
SUPPORTED_METHODS = AUDIO_SSL_METHODS | {"dinomaly"}
CHECKPOINT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="paper_benchmark/config.dcase2026_ssl.example.json"
    )
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--machines", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=13711)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ssl-epochs", type=int, default=None)
    parser.add_argument("--ssl-batch-size", type=int, default=None)
    parser.add_argument("--ssl-learning-rate", type=float, default=None)
    parser.add_argument("--ssl-device", default=None)
    parser.add_argument("--ssl-channel", type=int, default=None)
    parser.add_argument("--detector-channel", type=int, default=None)
    parser.add_argument("--ssl-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--force-ssl-pretrain", action="store_true")
    parser.add_argument(
        "--skip-ssl-pretrain",
        action="store_true",
        help="Require and reuse existing SSL checkpoints.",
    )
    parser.add_argument("--ssl-only", action="store_true")
    parser.add_argument("--write-decisions", action="store_true")
    parser.add_argument("--decision-percentile", type=float, default=99.0)
    parser.add_argument(
        "--score-aggregation",
        choices=("max", "mean", "temporal_topk_mean"),
        default=None,
    )
    parser.add_argument("--score-topk", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _target_sample_rate(backbone: str) -> int:
    """Return the sample rate expected by the selected log-mel frontend."""
    return 48000 if backbone == "HTSAT-base" else 44100


def _ssl_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "device": config.get("device", "auto"),
        "pretrain_channel": 1,
        "detector_channel": 0,
        "epochs": 10,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "temperature": 0.1,
        "projection_dim": 128,
        "noise_std": 0.02,
        "gain_min": 0.8,
        "gain_max": 1.2,
        "time_mask_fraction": 0.1,
    }
    values.update(config.get("dcase2026_ssl", {}))
    overrides = {
        "epochs": args.ssl_epochs,
        "batch_size": args.ssl_batch_size,
        "learning_rate": args.ssl_learning_rate,
        "device": args.ssl_device,
        "pretrain_channel": args.ssl_channel,
        "detector_channel": args.detector_channel,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})

    for key in ("pretrain_channel", "detector_channel"):
        values[key] = int(values[key])
        if values[key] not in {0, 1}:
            raise ValueError(f"{key} must be 0 or 1")
    if int(values["epochs"]) <= 0:
        raise ValueError("SSL epochs must be positive")
    if int(values["batch_size"]) < 2:
        raise ValueError("SSL batch_size must be at least 2 for contrastive learning")
    if float(values["learning_rate"]) <= 0:
        raise ValueError("SSL learning_rate must be positive")
    if float(values["temperature"]) <= 0:
        raise ValueError("SSL temperature must be positive")
    if int(values["projection_dim"]) <= 0:
        raise ValueError("SSL projection_dim must be positive")
    return values


def _augment_waveforms(batch: torch.Tensor, ssl: dict[str, Any]) -> torch.Tensor:
    """Create a stochastic view without changing waveform length."""
    if batch.ndim != 2:
        raise ValueError(f"Expected waveform batch [B, samples], got {tuple(batch.shape)}")
    output = batch.clone()
    batch_size, samples = output.shape

    gain_min = float(ssl["gain_min"])
    gain_max = float(ssl["gain_max"])
    if not 0 < gain_min <= gain_max:
        raise ValueError("SSL gains must satisfy 0 < gain_min <= gain_max")
    gains = torch.empty(batch_size, 1, device=output.device).uniform_(
        gain_min, gain_max
    )
    output = output * gains

    noise_std = float(ssl["noise_std"])
    if noise_std < 0:
        raise ValueError("SSL noise_std cannot be negative")
    if noise_std:
        rms = output.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        output = output + torch.randn_like(output) * rms * noise_std

    max_mask = int(samples * float(ssl["time_mask_fraction"]))
    if max_mask < 0 or max_mask > samples:
        raise ValueError("SSL time_mask_fraction must be between 0 and 1")
    if max_mask:
        lengths = torch.randint(0, max_mask + 1, (batch_size,), device=output.device)
        starts = torch.randint(0, samples, (batch_size,), device=output.device)
        for index, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
            end = min(start + length, samples)
            output[index, start:end] = 0

    max_shift = max(1, samples // 20)
    shifts = torch.randint(-max_shift, max_shift + 1, (batch_size,)).tolist()
    return torch.stack(
        [torch.roll(waveform, shifts=shift, dims=0) for waveform, shift in zip(output, shifts)]
    )


def _nt_xent_loss(
    first: torch.Tensor, second: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Symmetric NT-Xent loss for two augmented views of each recording."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("NT-Xent inputs must have the same [batch, features] shape")
    if first.shape[0] < 2:
        raise ValueError("NT-Xent requires at least two recordings per batch")
    embeddings = F.normalize(torch.cat((first, second), dim=0), dim=1)
    logits = embeddings @ embeddings.T / float(temperature)
    logits.fill_diagonal_(float("-inf"))
    batch_size = first.shape[0]
    positives = (torch.arange(2 * batch_size, device=logits.device) + batch_size) % (
        2 * batch_size
    )
    return F.cross_entropy(logits, positives)


class _ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def _audio_embedding(extractor, waveforms: torch.Tensor) -> torch.Tensor:
    feature_maps = extractor(waveforms)
    if not feature_maps:
        raise RuntimeError("The audio encoder did not expose any feature map")
    feature_map = feature_maps[-1]
    if feature_map.ndim != 4:
        raise RuntimeError(
            f"Expected a 4D audio feature map, got {tuple(feature_map.shape)}"
        )
    return F.adaptive_avg_pool2d(feature_map, 1).flatten(1)


def _dinomaly_embedding(model, waveforms: torch.Tensor) -> torch.Tensor:
    image, _ = model._to_image(waveforms)
    features = model.encoder.forward_features(image)
    embedding = model.encoder.forward_head(features, pre_logits=True)
    if embedding.ndim > 2:
        embedding = embedding.flatten(start_dim=1)
    return embedding


def _checkpoint_path(root: Path, machine: str, family: str) -> Path:
    return root / machine / f"{family}_far_to_near.pt"


def _load_ssl_checkpoint(
    path: Path,
    *,
    family: str,
    machine: str,
    backbone: str,
    channel: int,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "version": CHECKPOINT_VERSION,
        "family": family,
        "machine": machine,
        "backbone": backbone,
        "pretrain_channel": channel,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Incompatible SSL checkpoint {path}: {mismatches}")
    if "encoder_state_dict" not in checkpoint:
        raise ValueError(f"Missing encoder_state_dict in SSL checkpoint: {path}")
    return checkpoint


def _save_ssl_checkpoint(
    path: Path,
    *,
    encoder: nn.Module,
    family: str,
    machine: str,
    backbone: str,
    ssl: dict[str, Any],
    sample_rate: int,
    seed: int,
    losses: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu() for key, value in encoder.state_dict().items()
    }
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "objective": "nt_xent",
            "family": family,
            "machine": machine,
            "backbone": backbone,
            "pretrain_channel": int(ssl["pretrain_channel"]),
            "sample_rate": sample_rate,
            "seed": seed,
            "ssl_config": dict(ssl),
            "losses": losses,
            "encoder_state_dict": state_dict,
        },
        path,
    )


def _train_contrastive(
    *,
    encoder: nn.Module,
    embedding_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    ssl: dict[str, Any],
    device: torch.device,
    debug: bool,
) -> list[float]:
    encoder.train()
    for parameter in encoder.parameters():
        parameter.requires_grad = True

    first_batch = next(iter(loader)).to(device)
    with torch.no_grad():
        embedding_dim = int(embedding_fn(first_batch[:2]).shape[1])
    projection = _ProjectionHead(embedding_dim, int(ssl["projection_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projection.parameters()),
        lr=float(ssl["learning_rate"]),
    )
    max_batches = 2 if debug else None
    epoch_count = 1 if debug else int(ssl["epochs"])
    epoch_losses = []

    for epoch in range(epoch_count):
        total_loss = 0.0
        steps = 0
        for batch_index, waveforms in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if waveforms.shape[0] < 2:
                continue
            waveforms = waveforms.to(device)
            first_view = _augment_waveforms(waveforms, ssl)
            second_view = _augment_waveforms(waveforms, ssl)
            first_projection = projection(embedding_fn(first_view))
            second_projection = projection(embedding_fn(second_view))
            loss = _nt_xent_loss(
                first_projection, second_projection, float(ssl["temperature"])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
        if steps == 0:
            raise RuntimeError("SSL training did not receive a batch with at least 2 files")
        mean_loss = total_loss / steps
        epoch_losses.append(mean_loss)
        print(f"[DCASE2026 SSL] epoch={epoch + 1}/{epoch_count} loss={mean_loss:.6f}")
    return epoch_losses


def _pretrain_audio_ssl(
    config: dict[str, Any],
    loader: DataLoader,
    ssl: dict[str, Any],
    device: torch.device,
    debug: bool,
) -> tuple[nn.Module, list[float]]:
    extractor = make_feature_extractor(config, device, frozen=False)
    extractor.train()
    losses = _train_contrastive(
        encoder=extractor.model,
        embedding_fn=lambda batch: _audio_embedding(extractor, batch),
        loader=loader,
        ssl=ssl,
        device=device,
        debug=debug,
    )
    return extractor.model, losses


def _pretrain_dinomaly_ssl(
    config: dict[str, Any],
    loader: DataLoader,
    ssl: dict[str, Any],
    device: torch.device,
    debug: bool,
) -> tuple[nn.Module, list[float]]:
    model = make_model("dinomaly", config, device, (224, 224), "ssl")
    model.encoder.train()
    losses = _train_contrastive(
        encoder=model.encoder,
        embedding_fn=lambda batch: _dinomaly_embedding(model, batch),
        loader=loader,
        ssl=ssl,
        device=device,
        debug=debug,
    )
    return model.encoder, losses


def _prepare_ssl_checkpoint(
    *,
    family: str,
    machine: str,
    config: dict[str, Any],
    dataset_path: Path,
    checkpoint_root: Path,
    ssl: dict[str, Any],
    device: torch.device,
    seed: int,
    debug: bool,
    force: bool,
    skip_pretrain: bool,
) -> Path:
    backbone = (
        config.get("dinomaly_encoder", "deit_tiny_patch16_224.fb_in1k")
        if family == "dinomaly"
        else config["backbone"]
    )
    path = _checkpoint_path(checkpoint_root, machine, family)
    if path.exists() and not force:
        _load_ssl_checkpoint(
            path,
            family=family,
            machine=machine,
            backbone=backbone,
            channel=int(ssl["pretrain_channel"]),
        )
        print(f"[DCASE2026 SSL] reusing {path}")
        return path
    if skip_pretrain:
        raise FileNotFoundError(f"Required SSL checkpoint not found: {path}")

    sample_rate = _target_sample_rate(config["backbone"])
    dataset = DCASE2026Task2Dataset(
        dataset_path,
        machine,
        "train",
        train_domains="source",
        channel=int(ssl["pretrain_channel"]),
        target_sample_rate=sample_rate,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(ssl["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        drop_last=False,
    )
    set_seed(seed)
    if family == "audio":
        encoder, losses = _pretrain_audio_ssl(config, loader, ssl, device, debug)
    else:
        encoder, losses = _pretrain_dinomaly_ssl(config, loader, ssl, device, debug)
    _save_ssl_checkpoint(
        path,
        encoder=encoder,
        family=family,
        machine=machine,
        backbone=backbone,
        ssl=ssl,
        sample_rate=sample_rate,
        seed=seed,
        losses=losses,
    )
    print(f"[DCASE2026 SSL] saved {path}")
    return path


def _load_encoder_into_model(
    model,
    method: str,
    checkpoint: dict[str, Any],
) -> None:
    state_dict = checkpoint["encoder_state_dict"]
    if method == "dinomaly":
        model.encoder.load_state_dict(state_dict, strict=True)
        return

    if method == "stfpm":
        extractors = [model.teacher, model.student]
    elif method == "padim":
        extractors = [model.backbone_model]
    else:
        extractors = [model.feature_extractor]
    seen = set()
    for extractor in extractors:
        if id(extractor.model) in seen:
            continue
        extractor.model.load_state_dict(state_dict, strict=True)
        seen.add(id(extractor.model))


def _safe_metrics(
    scores: np.ndarray, labels: np.ndarray, domains: np.ndarray
) -> dict[str, float]:
    result: dict[str, float] = {}
    if np.unique(labels).size == 2:
        result["auc_all"] = float(metrics.roc_auc_score(labels, scores))
        result["pauc_all"] = float(
            metrics.roc_auc_score(labels, scores, max_fpr=0.1)
        )
    for name, domain in (("source", 0), ("target", 1)):
        mask = domains == domain
        if mask.any() and np.unique(labels[mask]).size == 2:
            result[f"auc_{name}"] = float(
                metrics.roc_auc_score(labels[mask], scores[mask])
            )
    return result


def _write_scored_ground_truth(
    dataset: DCASE2026Task2Dataset,
    test_paths: list[str],
    evaluator_root: Path,
    machine: str,
) -> None:
    """Write ground truth matching exactly the scored files, including debug runs."""
    records = {record.path.name: record for record in dataset.records}
    selected = [records[Path(path).name] for path in test_paths]
    names = [record.path.name for record in selected]
    labels = [record.label for record in selected]
    domains = [0 if record.domain == "source" else 1 for record in selected]
    attributes = [record.path.stem for record in selected]
    destinations = {
        "ground_truth_data": labels,
        "ground_truth_domain": domains,
        "ground_truth_attributes": attributes,
    }
    for directory, values in destinations.items():
        _write_pairs(
            evaluator_root
            / directory
            / f"ground_truth_{machine}_section_00_test.csv",
            names,
            values,
        )


def _run_detector(
    *,
    method: str,
    machine: str,
    config: dict[str, Any],
    dataset_path: Path,
    checkpoint_path: Path,
    ssl: dict[str, Any],
    args: argparse.Namespace,
    evaluator_root: Path,
) -> dict[str, Any]:
    device = resolve_device(config.get(f"{method}_device", config["device"]))
    sample_rate = _target_sample_rate(config["backbone"])
    train_dataset = DCASE2026Task2Dataset(
        dataset_path,
        machine,
        "train",
        train_domains="source",
        channel=int(ssl["detector_channel"]),
        target_sample_rate=sample_rate,
    )
    test_dataset = DCASE2026Task2Dataset(
        dataset_path,
        machine,
        "test",
        channel=int(ssl["detector_channel"]),
        target_sample_rate=sample_rate,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
    )
    train_score_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )
    spectrogram = make_spectrogram_transform(config).to(device)
    with torch.no_grad():
        sample = train_dataset[0].unsqueeze(0).to(device)
        input_size = tuple(int(value) for value in spectrogram(sample).shape[-2:])
    model = make_model(method, config, device, input_size, class_name=machine)
    family = "dinomaly" if method == "dinomaly" else "audio"
    backbone = (
        config.get("dinomaly_encoder", "deit_tiny_patch16_224.fb_in1k")
        if family == "dinomaly"
        else config["backbone"]
    )
    checkpoint = _load_ssl_checkpoint(
        checkpoint_path,
        family=family,
        machine=machine,
        backbone=backbone,
        channel=int(ssl["pretrain_channel"]),
    )
    _load_encoder_into_model(model, method, checkpoint)
    set_seed(args.seed)
    fit_model(
        method,
        model,
        train_loader,
        test_loader,
        config,
        device,
        args.debug,
        evaluate_during_training=False,
    )

    max_batches = int(config.get("debug_max_batches", 2)) if args.debug else None
    dcase = config.get("dcase2026", {})
    aggregation = args.score_aggregation or dcase.get(
        "score_aggregation", "temporal_topk_mean"
    )
    top_k = args.score_topk if args.score_topk is not None else int(
        dcase.get("score_topk", 5)
    )
    train_scores, _, train_paths = _score_model(
        model,
        train_score_loader,
        device,
        max_batches,
        aggregation,
        top_k,
    )
    test_scores, labels, test_paths = _score_model(
        model, test_loader, device, max_batches, aggregation, top_k
    )
    record_by_name = {record.path.name: record for record in test_dataset.records}
    domains = np.asarray(
        [0 if record_by_name[Path(path).name].domain == "source" else 1 for path in test_paths]
    )
    result: dict[str, Any] = _safe_metrics(test_scores, labels, domains)

    threshold = None
    decisions = None
    if args.write_decisions:
        threshold, decisions = percentile_decisions(
            test_scores, args.decision_percentile
        )
    result.update(
        {
            "method": method,
            "machine": machine,
            "protocol": "far_to_near",
            "train_domain": "source",
            "ssl_channel": int(ssl["pretrain_channel"]),
            "detector_channel": int(ssl["detector_channel"]),
            "ssl_checkpoint": str(checkpoint_path),
            "threshold": threshold,
            "threshold_source": "test" if threshold is not None else None,
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "score_aggregation": aggregation,
            "score_topk": top_k,
        }
    )

    team_dir = evaluator_root / "teams" / "moviad_ssl" / method
    _write_pairs(
        team_dir / f"train_anomaly_score_{machine}_section_00_train.csv",
        train_paths,
        train_scores,
    )
    _write_pairs(
        team_dir / f"anomaly_score_{machine}_section_00_test.csv",
        test_paths,
        test_scores,
    )
    if decisions is not None:
        _write_pairs(
            team_dir / f"decision_result_{machine}_section_00_test.csv",
            test_paths,
            decisions,
        )
    _write_scored_ground_truth(test_dataset, test_paths, evaluator_root, machine)
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.no_pretrained:
        config["pretrained"] = False
        config["dinomaly_pretrained"] = False
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.debug:
        config["memory_bank_size"] = min(
            int(config.get("memory_bank_size", 30000)), 64
        )

    methods = [method.lower() for method in (args.methods or config["methods"])]
    unsupported = sorted(set(methods) - SUPPORTED_METHODS)
    if unsupported:
        raise ValueError(f"Unsupported methods: {unsupported}")
    check_audio_checkpoint(config, methods)
    ssl = _ssl_config(config, args)
    configured_path = config.get("dcase2026", {}).get("dataset_path")
    dataset_path_value = args.dataset_path or configured_path
    if not dataset_path_value:
        raise ValueError(
            "Missing dataset path: set dcase2026.dataset_path or --dataset-path"
        )
    dataset_path = expand_path(dataset_path_value)
    machines = args.machines or discover_machine_types(dataset_path)
    if args.debug:
        machines = machines[:1]

    checkpoint_root = args.ssl_checkpoint_dir
    if checkpoint_root is None:
        configured_checkpoint_root = ssl.get("checkpoint_dir")
        checkpoint_root = (
            expand_path(configured_checkpoint_root)
            if configured_checkpoint_root
            else output_dir(config) / "dcase2026_ssl" / "checkpoints"
        )
    else:
        checkpoint_root = args.ssl_checkpoint_dir.expanduser().resolve()
    evaluator_root = (
        Path(__file__).resolve().parents[1]
        / "dcase2026_task2_evaluator"
        / "dev_ssl"
    )

    rows = []
    for machine in machines:
        checkpoints: dict[str, Path] = {}
        families = []
        if any(method in AUDIO_SSL_METHODS for method in methods):
            families.append("audio")
        if "dinomaly" in methods:
            families.append("dinomaly")
        for family in families:
            device = resolve_device(str(ssl["device"]))
            checkpoints[family] = _prepare_ssl_checkpoint(
                family=family,
                machine=machine,
                config=config,
                dataset_path=dataset_path,
                checkpoint_root=checkpoint_root,
                ssl=ssl,
                device=device,
                seed=args.seed,
                debug=args.debug,
                force=args.force_ssl_pretrain,
                skip_pretrain=args.skip_ssl_pretrain,
            )
        if args.ssl_only:
            continue
        for method in methods:
            family = "dinomaly" if method == "dinomaly" else "audio"
            print(f"[DCASE2026 SSL] method={method} machine={machine}")
            rows.append(
                _run_detector(
                    method=method,
                    machine=machine,
                    config=config,
                    dataset_path=dataset_path,
                    checkpoint_path=checkpoints[family],
                    ssl=ssl,
                    args=args,
                    evaluator_root=evaluator_root,
                )
            )

    if not args.ssl_only:
        result_path = output_dir(config) / "dcase2026_task2_ssl_results.json"
        result_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    run_cli(main)
