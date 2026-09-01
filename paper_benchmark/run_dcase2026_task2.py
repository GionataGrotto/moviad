"""Train/evaluate the existing audio AD models on DCASE 2026 Task 2 dev data.

This script uses the near microphone (channel 0), trains one model per machine,
and writes the two CSV files consumed by the vendored DCASE evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

try:
    from .benchmark_common import (
        check_audio_checkpoint,
        expand_path,
        fit_model,
        load_config,
        make_model,
        make_spectrogram_transform,
        output_dir,
        resolve_device,
        run_cli,
        set_seed,
    )
except ImportError:  # supports ``python paper_benchmark/run_*.py``
    from benchmark_common import (
        check_audio_checkpoint,
        expand_path,
        fit_model,
        load_config,
        make_model,
        make_spectrogram_transform,
        output_dir,
        resolve_device,
        run_cli,
        set_seed,
    )
from moviad.datasets.dcase2026_task2 import (
    DCASE2026Task2Dataset,
    discover_machine_types,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional override; normally use dcase2026.dataset_path in the config.",
    )
    parser.add_argument("--config", default="paper_benchmark/config.example.json")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--machines", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=13711)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--write-decisions",
        action="store_true",
        help="Also write binary decisions using --decision-percentile.",
    )
    parser.add_argument("--decision-percentile", type=float, default=99.0)
    parser.add_argument(
        "--score-aggregation",
        choices=("max", "mean", "temporal_topk_mean"),
        default=None,
        help="DCASE-only file score aggregation; defaults to dcase2026.score_aggregation.",
    )
    parser.add_argument(
        "--score-topk",
        type=int,
        default=None,
        help="Number of frequency bins for temporal_topk_mean (default: 5).",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Use random audio features for pipeline smoke tests")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _aggregate_dcase_score(output, aggregation: str, top_k: int = 5) -> np.ndarray:
    """Convert a model output into one continuous score per DCASE file.

    ``temporal_topk_mean`` follows the paper's temporal pooling idea: for each
    time position, average the top-k frequency responses and then average over
    time.  This function is deliberately used only by the DCASE runner;
    MIMII keeps its original evaluator/model score path.
    """
    if not isinstance(output, (tuple, list)):
        if aggregation != "max":
            raise ValueError("A score map is required for the selected aggregation")
        score = output
        if isinstance(score, torch.Tensor):
            return score.detach().cpu().reshape(-1).numpy()
        return np.asarray(score).reshape(-1)

    if aggregation == "max":
        score = output[1]
        if isinstance(score, torch.Tensor):
            return score.detach().cpu().reshape(-1).numpy()
        return np.asarray(score).reshape(-1)

    anomaly_map = output[0]
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.detach().cpu().numpy()
    anomaly_map = np.asarray(anomaly_map, dtype=float)
    if anomaly_map.ndim == 4:
        # Models expose a singleton output channel.  If a future model emits
        # multiple channels, average them before pooling.
        anomaly_map = anomaly_map.mean(axis=1)
    if anomaly_map.ndim != 3:
        raise ValueError(
            f"Expected an anomaly map with shape (batch,time,frequency), got {anomaly_map.shape}"
        )

    if aggregation == "mean":
        return anomaly_map.mean(axis=(1, 2))
    if aggregation == "temporal_topk_mean":
        if top_k <= 0:
            raise ValueError(f"score_topk must be positive, got {top_k}")
        top_k = min(top_k, anomaly_map.shape[2])
        top_frequency = np.partition(
            anomaly_map, anomaly_map.shape[2] - top_k, axis=2
        )[:, :, -top_k:]
        return top_frequency.mean(axis=(1, 2))
    raise ValueError(f"Unsupported DCASE score aggregation: {aggregation}")


def _score_model(
    model,
    loader,
    device,
    max_batches: int | None = None,
    aggregation: str = "temporal_topk_mean",
    top_k: int = 5,
):
    scores, labels, paths = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if isinstance(batch, torch.Tensor):
                waveform = batch
                label = None
                batch_paths = None
            elif len(batch) == 1:
                waveform = batch[0]
                label = None
                batch_paths = None
            else:
                waveform, label, batch_paths = batch
            output = model(waveform.to(device))
            score = _aggregate_dcase_score(output, aggregation, top_k)
            score = score.tolist()
            scores.extend(score)
            if label is not None:
                labels.extend(label.reshape(-1).tolist())
            if batch_paths is not None:
                paths.extend(batch_paths)
            else:
                start = len(paths)
                paths.extend(f"train_{start + index:06d}.wav" for index in range(len(score)))
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=int), paths


def _write_pairs(path: Path, filenames, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for filename, value in sorted(zip(filenames, values)):
            writer.writerow([Path(filename).name, value])


def _write_dev_ground_truth(dataset_path: Path, evaluator_root: Path, machine: str) -> None:
    dataset = DCASE2026Task2Dataset(dataset_path, machine, "test")
    names = [record.path.name for record in dataset.records]
    labels = [record.label for record in dataset.records]
    domains = [0 if record.domain == "source" else 1 for record in dataset.records]
    for directory in ("ground_truth_data", "ground_truth_domain"):
        evaluator_root.joinpath(directory).mkdir(parents=True, exist_ok=True)
    _write_pairs(evaluator_root / "ground_truth_data" / f"ground_truth_{machine}_section_00_test.csv", names, labels)
    _write_pairs(evaluator_root / "ground_truth_domain" / f"ground_truth_{machine}_section_00_test.csv", names, domains)
    # The upstream evaluator requires an attributes file even when attributes
    # are hidden. It is only used to remap baseline-style filenames.
    _write_pairs(evaluator_root / "ground_truth_attributes" / f"ground_truth_{machine}_section_00_test.csv", names, [Path(n).stem for n in names])


def _metrics(scores, labels, domains):
    result = {
        "auc_all": metrics.roc_auc_score(labels, scores),
        "pauc_all": metrics.roc_auc_score(labels, scores, max_fpr=0.1),
    }
    for name, domain in (("source", 0), ("target", 1)):
        mask = domains == domain
        result[f"auc_{name}"] = metrics.roc_auc_score(labels[mask], scores[mask])
    return result


def run_one(method, config, dataset_path, machine, seed, args, evaluator_root):
    device = resolve_device(config.get(f"{method}_device", config["device"]))
    set_seed(seed)
    train_ds = DCASE2026Task2Dataset(dataset_path, machine, "train", train_domains="all", channel=0)
    test_ds = DCASE2026Task2Dataset(dataset_path, machine, "test", channel=0)
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config["num_workers"]))
    test_loader = DataLoader(test_ds, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]))
    spectro = make_spectrogram_transform(config)
    input_size = tuple(int(dim) for dim in spectro(train_ds[0].unsqueeze(0)).shape[-2:])
    model = make_model(method, config, device, input_size, class_name=machine)
    if args.epochs is not None:
        config["epochs"] = args.epochs
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
    dcase_config = config.get("dcase2026", {})
    aggregation = args.score_aggregation or dcase_config.get(
        "score_aggregation", "temporal_topk_mean"
    )
    top_k = (
        args.score_topk
        if args.score_topk is not None
        else int(dcase_config.get("score_topk", 5))
    )
    train_scores, _, train_paths = _score_model(
        model, train_loader, device, max_batches, aggregation, top_k
    )
    test_scores, labels, test_paths = _score_model(
        model, test_loader, device, max_batches, aggregation, top_k
    )
    threshold = None
    decisions = None
    if args.write_decisions:
        threshold = float(np.percentile(train_scores, args.decision_percentile))
        decisions = (test_scores >= threshold).astype(int)
    domains = np.asarray([0 if record.domain == "source" else 1 for record in test_ds.records])
    result = _metrics(test_scores, labels, domains) if len(np.unique(labels)) == 2 else {}
    result.update({"method": method, "machine": machine, "threshold": threshold, "train_size": len(train_ds), "test_size": len(test_ds), "score_aggregation": aggregation, "score_topk": top_k})

    team_dir = evaluator_root / "teams" / "moviad" / method
    _write_pairs(
        team_dir / f"train_anomaly_score_{machine}_section_00_train.csv",
        train_paths,
        train_scores,
    )
    _write_pairs(team_dir / f"anomaly_score_{machine}_section_00_test.csv", test_paths, test_scores)
    if decisions is not None:
        _write_pairs(team_dir / f"decision_result_{machine}_section_00_test.csv", test_paths, decisions)
    _write_dev_ground_truth(dataset_path, evaluator_root, machine)
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.no_pretrained:
        config["pretrained"] = False
    if args.debug:
        # PatchCore's default 30k coreset is appropriate for a full run but
        # makes a one-machine CPU smoke test needlessly expensive.
        config["memory_bank_size"] = min(int(config.get("memory_bank_size", 30000)), 64)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    methods = [m.lower() for m in (args.methods or config["methods"])]
    check_audio_checkpoint(config, methods)
    configured_dataset_path = config.get("dcase2026", {}).get("dataset_path")
    dataset_path_value = args.dataset_path or configured_dataset_path
    if not dataset_path_value:
        raise ValueError(
            "Missing DCASE dataset path. Set dcase2026.dataset_path in the config "
            "or pass --dataset-path as an override."
        )
    dataset_path = expand_path(dataset_path_value)
    # Keep development ground truth/results separate from the upstream
    # evaluation-set files shipped by the evaluator repository.
    evaluator_root = Path(__file__).resolve().parents[1] / "dcase2026_task2_evaluator" / "dev"
    machines = args.machines or discover_machine_types(dataset_path)
    if args.debug:
        machines = machines[:1]
    rows = []
    for method in methods:
        for machine in machines:
            print(f"[DCASE2026] method={method} machine={machine}")
            rows.append(run_one(method, config, dataset_path, machine, args.seed, args, evaluator_root))
    result_path = output_dir(config) / "dcase2026_task2_results.json"
    result_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    run_cli(main)
