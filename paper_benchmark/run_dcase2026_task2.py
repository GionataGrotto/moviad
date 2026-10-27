"""Train/evaluate the existing audio AD models on DCASE 2026 Task 2 dev data.

This script uses the near microphone (channel 0), trains one model per machine,
and writes the continuous score CSVs consumed by the vendored DCASE evaluator.
Decision CSVs are optional and can be generated with ``--write-decisions``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

try:
    from .benchmark_common import (
        check_audio_checkpoint,
        method_sample_rate,
        prepare_audio_run,
        expand_path,
        fit_model,
        load_config,
        make_model,
        spectrogram_size,
        optimal_f1_metrics,
        output_dir,
        percentile_decisions,
        resolve_device,
        run_cli,
        set_seed,
    )
except ImportError:  # supports ``python paper_benchmark/run_*.py``
    from benchmark_common import (
        check_audio_checkpoint,
        method_sample_rate,
        prepare_audio_run,
        expand_path,
        fit_model,
        load_config,
        make_model,
        spectrogram_size,
        optimal_f1_metrics,
        output_dir,
        percentile_decisions,
        resolve_device,
        run_cli,
        set_seed,
    )
from moviad.datasets.dcase2026_task2 import (
    DCASE2026Task2Dataset,
    discover_machine_types,
)


_TOP_PERCENT = re.compile(r"^top(\d+(?:\.\d+)?)pct$")
# "frames<N>pct": per frame the mean of the top-k frequency responses (as temporal_topk_mean), then the mean of the
# highest N% of the frames instead of all of them
_FRAMES_PERCENT = re.compile(r"^frames(\d+(?:\.\d+)?)pct$")
_BASIC_AGGREGATIONS = ("max", "mean", "temporal_topk_mean")


def validate_aggregation(name: str) -> str:
    """``max``, ``mean``, ``temporal_topk_mean`` or ``top<N>pct`` with 0 < N <= 100."""
    if name in _BASIC_AGGREGATIONS:
        return name
    match = _TOP_PERCENT.match(name) or _FRAMES_PERCENT.match(name)
    if match and 0 < float(match.group(1)) <= 100:
        return name
    raise argparse.ArgumentTypeError(
        f"Unsupported DCASE score aggregation {name!r}: use max, mean, temporal_topk_mean top<N>pct or frames<N>pct (0 < N <= 100)"
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
        help="Also write binary decisions using a percentile of the test scores.",
    )
    parser.add_argument(
        "--decision-percentile",
        type=float,
        default=99.0,
        help="Percentile of the test scores used for binary decisions (default: 99).",
    )
    parser.add_argument(
        "--score-aggregation",
        type=validate_aggregation,
        default=None,
        help="DCASE-only file score aggregation: max, mean, temporal_topk_mean or top<N>pct "
        "(mean of the highest N%% of the map, e.g. top5pct); defaults to dcase2026.score_aggregation.",
    )
    parser.add_argument(
        "--extra-aggregations",
        nargs="+",
        type=validate_aggregation,
        default=None,
        help="More aggregations computed from the same trained model and written next to the main one "
        "as <method>@<aggregation>; defaults to dcase2026.extra_aggregations.",
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
    match = _TOP_PERCENT.match(aggregation)
    if match:
        # mean of the highest N% values of the whole map: between "max" (one patch) and "mean" (all)
        values = anomaly_map.reshape(anomaly_map.shape[0], -1)
        count = max(1, math.ceil(float(match.group(1)) / 100 * values.shape[1]))
        return np.partition(values, values.shape[1] - count, axis=1)[:, -count:].mean(axis=1)
    frames = _FRAMES_PERCENT.match(aggregation)
    if aggregation == "temporal_topk_mean" or frames:
        if top_k <= 0:
            raise ValueError(f"score_topk must be positive, got {top_k}")
        top_k = min(top_k, anomaly_map.shape[2])
        top_frequency = np.partition(
            anomaly_map, anomaly_map.shape[2] - top_k, axis=2
        )[:, :, -top_k:]
        if not frames:
            return top_frequency.mean(axis=(1, 2))
        per_frame = top_frequency.mean(axis=2)  # (batch, time)
        count = max(1, math.ceil(float(frames.group(1)) / 100 * per_frame.shape[1]))
        return np.partition(per_frame, per_frame.shape[1] - count, axis=1)[:, -count:].mean(axis=1)
    raise ValueError(f"Unsupported DCASE score aggregation: {aggregation}")


def _score_model_multi(
    model,
    loader,
    device,
    max_batches: int | None = None,
    aggregations: tuple[str, ...] = ("temporal_topk_mean",),
    top_k: int = 5,
):
    """Score a loader once and aggregate every output with each of ``aggregations``.

    Returns ``({aggregation: scores}, labels, paths)``; the forward pass, which is the costly part,
    is shared by all aggregations.
    """
    scores = {aggregation: [] for aggregation in aggregations}
    labels, paths = [], []
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
            for aggregation in aggregations:
                batch_scores = _aggregate_dcase_score(output, aggregation, top_k).tolist()
                scores[aggregation].extend(batch_scores)
            count = len(batch_scores)
            if label is not None:
                labels.extend(label.reshape(-1).tolist())
            if batch_paths is not None:
                paths.extend(batch_paths)
            else:
                start = len(paths)
                paths.extend(f"train_{start + index:06d}.wav" for index in range(count))
    return {a: np.asarray(v, dtype=float) for a, v in scores.items()}, np.asarray(labels, dtype=int), paths


def _score_model(
    model,
    loader,
    device,
    max_batches: int | None = None,
    aggregation: str = "temporal_topk_mean",
    top_k: int = 5,
):
    scores, labels, paths = _score_model_multi(model, loader, device, max_batches, (aggregation,), top_k)
    return scores[aggregation], labels, paths


def _write_pairs(path: Path, filenames, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(filenames) != len(values):
        raise ValueError("Filenames and values must have matching lengths")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for filename, value in sorted(zip(filenames, values)):
            writer.writerow([Path(filename).name, value])


def _write_dev_ground_truth(dataset_path: Path, evaluator_root: Path, machine: str, scored_paths=None) -> None:
    dataset = DCASE2026Task2Dataset(dataset_path, machine, "test")
    if scored_paths is not None:
        selected = {Path(path).name for path in scored_paths}
        dataset.records = [record for record in dataset.records if record.path.name in selected]
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
    # Threshold-free, prevalence-independent complement to the DCASE
    # evaluator's percentile-calibrated precision/recall/F1: see
    # optimal_f1_metrics's docstring for why the two are not interchangeable
    # and when each is the right one to read.
    for key, value in optimal_f1_metrics(scores, labels).items():
        result[f"{key}_all"] = value
    for name, domain in (("source", 0), ("target", 1)):
        mask = domains == domain
        result[f"auc_{name}"] = (
            metrics.roc_auc_score(labels[mask], scores[mask])
            if len(np.unique(labels[mask])) == 2 else None
        )
        if len(np.unique(labels[mask])) == 2:
            for key, value in optimal_f1_metrics(scores[mask], labels[mask]).items():
                result[f"{key}_{name}"] = value
    return result


def load_results(path: Path) -> list[dict]:
    """Rows saved by earlier invocations of the same run (empty if there are none or the file is unreadable)."""
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        path.rename(path.with_suffix(".corrupt.json"))  # keep the broken file instead of silently dropping it
        return []
    return rows if isinstance(rows, list) else []


def upsert_result(rows: list[dict], row: dict) -> list[dict]:
    """Replace the row of the same (method, machine, debug) or append a new one.

    The file used to be rewritten from scratch by every invocation, so repeating or extending a run
    (another method, a missing machine) erased the rows of the previous ones.
    """
    key = (row.get("method"), row.get("machine"), row.get("debug"))
    kept = [r for r in rows if (r.get("method"), r.get("machine"), r.get("debug")) != key]
    return kept + [row]


def run_one(method, config, dataset_path, machine, seed, args, evaluator_root):
    device = resolve_device(config.get(f"{method}_device", config["device"]))
    set_seed(seed)
    dcase = config.get("dcase2026", {})
    channel = dcase.get("channel", 0)
    channel = channel if channel in ("diff", "ls", "ls_half", "ls_win", "ls_global") else int(channel)
    train_domain = dcase.get("train_domains", "all")
    test_domain = dcase.get("test_domain", "all")
    sample_rate = method_sample_rate(method, config)
    train_ds = DCASE2026Task2Dataset(dataset_path, machine, "train", train_domains=train_domain, channel=channel, target_sample_rate=sample_rate)
    test_ds = DCASE2026Task2Dataset(dataset_path, machine, "test", domain=test_domain, channel=channel, target_sample_rate=sample_rate)
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config["num_workers"]))
    test_loader = DataLoader(test_ds, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]))
    input_size = spectrogram_size(config, train_ds[0])
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
    extra_aggregations = [
        a for a in (args.extra_aggregations or dcase_config.get("extra_aggregations", [])) if a != aggregation
    ]
    for name in extra_aggregations:
        validate_aggregation(name)
    all_aggregations = (aggregation, *dict.fromkeys(extra_aggregations))
    train_score_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]))
    train_by_aggregation, _, _ = _score_model_multi(
        model, train_score_loader, device, max_batches, all_aggregations, top_k
    )
    train_scores = train_by_aggregation[aggregation]
    train_paths = [str(record.path) for record in train_ds.records[:len(train_scores)]]
    test_by_aggregation, labels, test_paths = _score_model_multi(
        model, test_loader, device, max_batches, all_aggregations, top_k
    )
    test_scores = test_by_aggregation[aggregation]
    threshold = None
    decisions = None
    if args.write_decisions:
        threshold, decisions = percentile_decisions(
            test_scores, args.decision_percentile
        )
    records = {str(record.path): record for record in test_ds.records}
    domains = np.asarray([0 if records[path].domain == "source" else 1 for path in test_paths])
    result = _metrics(test_scores, labels, domains) if len(np.unique(labels)) == 2 else {}
    result.update({
        "method": method,
        "seed": seed,
        "channel": channel,
        "sample_rate": sample_rate,
        "train_domain": train_domain,
        "test_domain": test_domain,
        "debug": args.debug,
        "machine": machine,
        "threshold": threshold,
        "threshold_source": "test" if threshold is not None else None,
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "score_aggregation": aggregation,
        "score_topk": top_k,
        "backbone": config.get("backbone"),
        "layers": config.get("layers"),
        "run_name": dcase.get("run_name"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    if len(all_aggregations) > 1 and len(np.unique(labels)) == 2:
        result["extra_aggregations"] = {
            name: _metrics(test_by_aggregation[name], labels, domains) for name in all_aggregations[1:]
        }

    team_dir = evaluator_root / "teams" / "moviad" / method
    _write_pairs(
        team_dir / f"train_anomaly_score_{machine}_section_00_train.csv",
        train_paths,
        train_scores,
    )
    _write_pairs(team_dir / f"anomaly_score_{machine}_section_00_test.csv", test_paths, test_scores)
    if decisions is not None:
        _write_pairs(team_dir / f"decision_result_{machine}_section_00_test.csv", test_paths, decisions)
    for name in all_aggregations[1:]:
        extra_dir = evaluator_root / "teams" / "moviad" / f"{method}@{name}"
        _write_pairs(
            extra_dir / f"train_anomaly_score_{machine}_section_00_train.csv", train_paths, train_by_aggregation[name]
        )
        _write_pairs(extra_dir / f"anomaly_score_{machine}_section_00_test.csv", test_paths, test_by_aggregation[name])
    _write_dev_ground_truth(dataset_path, evaluator_root, machine, test_paths)
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
    evaluator_root = prepare_audio_run(config, args, "dev")
    machines = args.machines or discover_machine_types(dataset_path)
    if args.debug:
        machines = machines[:1]
    result_path = output_dir(config) / "dcase2026_task2_results.json"
    rows = load_results(result_path)
    for method in methods:
        for machine in machines:
            print(f"[DCASE2026] method={method} machine={machine}")
            row = run_one(method, config, dataset_path, machine, args.seed, args, evaluator_root)
            rows = upsert_result(rows, row)
            result_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    run_cli(main)
