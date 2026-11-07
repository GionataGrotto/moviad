"""How much of the evaluation score is lost to the score offset between the two domains? (a diagnostic, not a method)

The official AUC of a domain compares the normal clips of the domain with ALL the anomalous clips. If the normal clips of one
domain get systematically higher scores than those of the other, one domain is penalised even when the detector ranks the clips
inside each domain well. This script re-scores a run after removing the offset with the ORACLE domain labels of the test clips
(it uses the evaluation ground truth, so it is an upper bound of what a domain-aware normalisation could give, not a result):

    shift   score - median of the normal clips of the clip's domain
    zscore  (score - median) / std of the normal clips of the clip's domain

    python paper_benchmark/domain_calibration.py --pairs tgt_pc_quota10:patchcore knn_ft_ls_both:knn_ft
    python paper_benchmark/domain_calibration.py --scores name=<folder with anomaly_score_*.csv>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fuse_scores import read_machine_scores
from score_eval import GT_ROOT, eval_machines, machine_metrics, read_ground_truth


def calibrate(scores: dict[str, float], labels: dict[str, int], domains: dict[str, int], mode: str) -> dict[str, float]:
    """Scores of one machine with the offset of each domain removed, estimated on the normal clips of that domain."""
    files = sorted(labels)
    s = np.array([scores[f] for f in files], dtype=float)
    y = np.array([labels[f] for f in files])
    d = np.array([domains[f] for f in files])
    out = s.copy()
    for flag in (0, 1):
        normal = (d == flag) & (y == 0)
        if normal.sum() < 2:
            continue
        centre = np.median(s[normal])
        scale = s[normal].std() if mode == "zscore" else 1.0
        out[d == flag] = (s[d == flag] - centre) / (scale if scale > 0 else 1.0)
    return dict(zip(files, out))


def offset_in_std(scores: dict[str, float], labels: dict[str, int], domains: dict[str, int]) -> float:
    """Median score of the target normals minus that of the source normals, in standard deviations of the source normals."""
    files = sorted(labels)
    s = np.array([scores[f] for f in files], dtype=float)
    y = np.array([labels[f] for f in files])
    d = np.array([domains[f] for f in files])
    source, target = s[(d == 0) & (y == 0)], s[(d == 1) & (y == 0)]
    return float((np.median(target) - np.median(source)) / (source.std() or 1.0))


def official(result: dict) -> float:
    from scipy.stats import hmean  # the evaluator's score: harmonic mean of source AUC, target AUC and pAUC of every machine

    values = ([m["auc_source_official"] for m in result.values()] + [m["auc_target_official"] for m in result.values()]
              + [m["pauc_all"] for m in result.values()])
    return float(hmean(np.maximum(np.array(values, dtype=float), 1e-12))) * 100


def report(name: str, scores: dict[str, dict[str, float]], gt_root: Path = GT_ROOT) -> str:
    machines = [m for m in eval_machines(gt_root) if m in scores]
    truth = {m: read_ground_truth(m, gt_root) for m in machines}
    row = {}
    for mode in ("raw", "shift", "zscore"):
        result = {m: machine_metrics(scores[m] if mode == "raw" else calibrate(scores[m], *truth[m], mode), *truth[m]) for m in machines}
        row[mode] = official(result)
    offsets = [offset_in_std(scores[m], *truth[m]) for m in machines]
    return (f"| {name} | {row['raw']:.1f} | {row['shift']:.1f} | {row['zscore']:.1f} | {row['zscore'] - row['raw']:+.1f} | "
            f"{np.mean(offsets):+.2f} ({' '.join(f'{o:+.1f}' for o in offsets)}) |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="*", default=[], metavar="EXPERIMENT:METHOD", help="evaluation runs (grid_<experiment>_eval)")
    parser.add_argument("--scores", nargs="*", default=[], metavar="NAME=FOLDER")
    args = parser.parse_args()
    inputs = {}
    for item in args.pairs:
        import run_eval

        pair = tuple(item.split(":", 1))
        inputs[item] = read_machine_scores(run_eval.run_folder(pair))
    for item in args.scores:
        name, _, folder = item.partition("=")
        inputs[name] = read_machine_scores(Path(folder).expanduser())
    print("| Run | Official score | With shift removed (oracle) | With z-score (oracle) | Gain | Offset target - source normals, in std (mean, per machine) |")
    print("|---|---|---|---|---|---|")
    for name, scores in inputs.items():
        print(report(name, scores))


if __name__ == "__main__":
    main()
