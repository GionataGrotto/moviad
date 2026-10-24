"""Fuse the anomaly scores of several runs (methods, seeds, encoders) into one score per clip.

Every input is a folder with ``anomaly_score_<machine>_section_00_test.csv`` files, as written by the
DCASE runner. For each machine the scores of every input are put on a common scale and averaged.
The default is the rank within the machine (the score of a clip becomes its position among the
clips of that machine, in [0, 1]), which does not depend on the very different score ranges of
PatchCore, CFA, kNN... Weights are equal unless given: choose them a priori, not by looking at the test.

    python paper_benchmark/fuse_scores.py --inputs patchcore=dir1 cfa=dir2 knn=dir3
    python paper_benchmark/fuse_scores.py --inputs a=dir1 b=dir2 --write fused_dir

Averaging runs of the same method with different seeds is the same operation and reduces the
run-to-run variance (the CFA varies by several points between seeds).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SCORE_FILE = re.compile(r"^anomaly_score_(?P<machine>.+)_section_(?P<section>\d+)_test\.csv$")


def read_machine_scores(folder: Path) -> dict[str, dict[str, float]]:
    """``{machine: {file name: score}}`` of a score folder."""
    machines = {}
    for path in sorted(Path(folder).glob("anomaly_score_*_test.csv")):
        match = SCORE_FILE.match(path.name)
        if match is None:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            machines[match["machine"]] = {row[0]: float(row[1]) for row in csv.reader(handle) if row}
    return machines


def normalise(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "rank":
        order = values.argsort(kind="stable")
        ranks = np.empty(len(values), dtype=float)
        # ties share the average of the positions they occupy
        sorted_values = values[order]
        position = 0
        while position < len(values):
            end = position
            while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[position]:
                end += 1
            ranks[order[position : end + 1]] = (position + end) / 2
            position = end + 1
        return ranks / max(len(values) - 1, 1)
    if mode == "zscore":
        std = values.std()
        return (values - values.mean()) / std if std > 0 else np.zeros_like(values)
    if mode == "none":
        return values
    raise ValueError(f"Unsupported normalisation {mode!r}: use rank, zscore or none")


def fuse(
    inputs: dict[str, dict[str, dict[str, float]]],
    weights: dict[str, float] | None = None,
    norm: str = "rank",
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    """Fuse ``{name: {machine: {file: score}}}``; machines missing from an input are skipped and reported."""
    names = list(inputs)
    weights = weights or {name: 1.0 for name in names}
    total = sum(weights[name] for name in names)
    machines = sorted(set().union(*(inputs[name].keys() for name in names)))
    fused, skipped = {}, {}
    for machine in machines:
        missing = [name for name in names if machine not in inputs[name]]
        if missing:
            skipped[machine] = missing
            continue
        files = sorted(inputs[names[0]][machine])
        for name in names:
            if sorted(inputs[name][machine]) != files:
                raise ValueError(f"The inputs score different clips for machine {machine!r} ({name!r} differs)")
        combined = np.zeros(len(files))
        for name in names:
            values = np.array([inputs[name][machine][file] for file in files])
            combined += weights[name] * normalise(values, norm)
        fused[machine] = dict(zip(files, combined / total))
    return fused, skipped


def machine_auc(scores: dict[str, float], domain: str = "source") -> tuple[float | None, float | None]:
    selected = [(file, score) for file, score in scores.items() if domain == "all" or f"_{domain}_" in file]
    labels = [1 if "_anomaly_" in file else 0 for file, _ in selected]
    if len(set(labels)) < 2:
        return None, None
    values = [score for _, score in selected]
    return roc_auc_score(labels, values), roc_auc_score(labels, values, max_fpr=0.1)


def mean_auc(machines: dict[str, dict[str, float]], domain: str = "source") -> tuple[float | None, int]:
    aucs = [auc for auc, _ in (machine_auc(scores, domain) for scores in machines.values()) if auc is not None]
    return (sum(aucs) / len(aucs) if aucs else None), len(aucs)


def write_fused(path: Path, fused: dict[str, dict[str, float]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for machine, scores in fused.items():
        with (path / f"anomaly_score_{machine}_section_00_test.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            for file, score in sorted(scores.items()):
                writer.writerow([file, score])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="name=folder pairs")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="one weight per input (default: equal)")
    parser.add_argument("--norm", choices=("rank", "zscore", "none"), default="rank")
    parser.add_argument("--domain", choices=("source", "target", "all"), default="source")
    parser.add_argument("--write", type=Path, default=None, help="write the fused scores to this folder")
    args = parser.parse_args()

    inputs = {}
    for item in args.inputs:
        name, _, folder = item.partition("=")
        if not folder:
            raise SystemExit(f"--inputs expects name=folder, got {item!r}")
        inputs[name] = read_machine_scores(Path(folder).expanduser())
    weights = None
    if args.weights is not None:
        if len(args.weights) != len(inputs):
            raise SystemExit("--weights needs one value per input")
        weights = dict(zip(inputs, args.weights))

    fused, skipped = fuse(inputs, weights, args.norm)
    for name, machines in inputs.items():
        mean, count = mean_auc({m: s for m, s in machines.items() if m in fused}, args.domain)
        print(f"{name:24s} AUC={mean:.4f}  (machines: {count})")
    mean, count = mean_auc(fused, args.domain)
    print(f"{'FUSED':24s} AUC={mean:.4f}  (machines: {count}, norm: {args.norm})")
    for machine, scores in fused.items():
        auc, _ = machine_auc(scores, args.domain)
        print(f"  {machine:12s} {auc:.4f}")
    for machine, missing in skipped.items():
        print(f"skipped {machine}: missing in {missing}")
    if args.write:
        write_fused(args.write, fused)
        print(f"fused scores written to {args.write}")


if __name__ == "__main__":
    main()
