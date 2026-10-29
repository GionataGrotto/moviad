"""AUC of score files on the DCASE 2026 Task 2 evaluation set, from the ground truth shipped with the evaluator.

The evaluation test clips are called ``section_00_0001.wav``..., so source / target and normal / anomaly come from
``dcase2026_task2_evaluator/ground_truth_domain`` (0 = source, 1 = target) and ``ground_truth_data`` (1 = anomaly).

    python paper_benchmark/score_eval.py --scores <run>/teams/moviad/patchcore ...
    python paper_benchmark/score_eval.py --scores a=<folder> b=<folder> --fuse      # rank-average of the inputs as well
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from fuse_scores import fuse, read_machine_scores

ROOT = Path(__file__).resolve().parent.parent
GT_ROOT = ROOT / "dcase2026_task2_evaluator"


def read_ground_truth(machine: str, gt_root: Path = GT_ROOT) -> tuple[dict[str, int], dict[str, int]]:
    """``({file: 1 if anomaly}, {file: 1 if target domain})`` of one evaluation machine."""
    def read(folder):
        with (gt_root / folder / f"ground_truth_{machine}_section_00_test.csv").open(newline="", encoding="utf-8") as handle:
            return {row[0]: int(row[1]) for row in csv.reader(handle) if row}

    return read("ground_truth_data"), read("ground_truth_domain")


def eval_machines(gt_root: Path = GT_ROOT) -> list[str]:
    """The machines that have evaluation ground truth (the dev machines' files are written under ``dev/`` instead)."""
    prefix, suffix = "ground_truth_", "_section_00_test.csv"
    return sorted(p.name[len(prefix) : -len(suffix)] for p in (gt_root / "ground_truth_data").glob(f"{prefix}*{suffix}"))


def machine_metrics(scores: dict[str, float], labels: dict[str, int], domains: dict[str, int]) -> dict[str, float | None]:
    missing = set(labels) - set(scores)
    if missing:
        raise ValueError(f"{len(missing)} test clips have no score (e.g. {sorted(missing)[0]})")
    files = sorted(labels)
    y = np.array([labels[f] for f in files])
    s = np.array([scores[f] for f in files])
    d = np.array([domains[f] for f in files])
    out = {"auc_all": roc_auc_score(y, s), "pauc_all": roc_auc_score(y, s, max_fpr=0.1)}
    for name, flag in (("source", 0), ("target", 1)):
        mask = d == flag
        out[f"auc_{name}"] = roc_auc_score(y[mask], s[mask]) if len(np.unique(y[mask])) == 2 else None
    return out


def evaluate(scores: dict[str, dict[str, float]], gt_root: Path = GT_ROOT) -> dict[str, dict]:
    """Metrics per machine (only the machines the scores cover)."""
    return {m: machine_metrics(scores[m], *read_ground_truth(m, gt_root)) for m in eval_machines(gt_root) if m in scores}


def mean_metric(per_machine: dict[str, dict], key: str = "auc_source") -> float | None:
    values = [m[key] for m in per_machine.values() if m.get(key) is not None]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", nargs="+", required=True, help="folder or name=folder")
    parser.add_argument("--fuse", action="store_true", help="also evaluate the rank-average of the inputs")
    args = parser.parse_args()
    inputs = {}
    for item in args.scores:
        name, sep, folder = item.partition("=")
        inputs[name if sep else Path(item).name] = read_machine_scores(Path(folder if sep else item).expanduser())
    if args.fuse and len(inputs) > 1:
        inputs["FUSED"] = fuse(dict(inputs))[0]
    for name, scores in inputs.items():
        result = evaluate(scores)
        print(f"{name:30s} AUC source={mean_metric(result):.3f}  all={mean_metric(result, 'auc_all'):.3f}  "
              f"target={mean_metric(result, 'auc_target') or float('nan'):.3f}  | "
              + " ".join(f"{m[:6]}={r['auc_source']:.2f}" for m, r in result.items()))


if __name__ == "__main__":
    main()
