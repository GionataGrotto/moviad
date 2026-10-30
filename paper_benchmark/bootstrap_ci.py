"""Bootstrap confidence intervals of the mean source AUC, and of the difference between two runs.

The clips of every machine are resampled with replacement (normal and anomalous clips separately, so that both
classes are always present), the AUC of each machine is recomputed and averaged over the machines. The same resamples
are used for all the runs, so the interval of a *difference* is paired and much tighter than two separate intervals.

    # dev machines (labels and domain come from the file names)
    python paper_benchmark/bootstrap_ci.py --dev --scores a=<run>/teams/moviad/patchcore b=<run2>/teams/moviad/patchcore
    # evaluation set (labels from the ground truth of the evaluator)
    python paper_benchmark/bootstrap_ci.py --eval --scores a=<folder> b=<folder>

The first input is the reference: for every other one the difference ``other - reference`` is printed with its 95%
interval and the fraction of resamples in which it is positive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from fuse_scores import read_machine_scores
from score_eval import GT_ROOT, eval_machines, read_ground_truth


def dev_labels(scores: dict[str, float], domain: str = "source") -> dict[str, int]:
    """Anomaly label of the clips of one domain, from the DCASE dev file names."""
    return {f: int("_anomaly_" in f) for f in scores if f"_{domain}_" in f}


def eval_labels(machine: str, gt_root: Path = GT_ROOT, domain: str = "source") -> dict[str, int]:
    labels, domains = read_ground_truth(machine, gt_root)
    wanted = 0 if domain == "source" else 1
    return {f: labels[f] for f in labels if domains[f] == wanted}


def bootstrap_means(scores: dict[str, dict[str, dict[str, float]]], labels: dict[str, dict[str, int]], n: int = 1000, seed: int = 0) -> dict[str, np.ndarray]:
    """``{run: n bootstrap values of the mean AUC over the machines}``; machines are those with labels and scores in every run."""
    machines = [m for m in labels if all(m in run for run in scores.values())]
    if not machines:
        raise ValueError("No machine has both labels and scores in every run")
    generator = np.random.default_rng(seed)
    result = {name: np.zeros(n) for name in scores}
    for machine in machines:
        files = sorted(labels[machine])
        y = np.array([labels[machine][f] for f in files])
        normal, anomalous = np.where(y == 0)[0], np.where(y == 1)[0]
        values = {name: np.array([run[machine][f] for f in files]) for name, run in scores.items()}
        for i in range(n):
            idx = np.concatenate([generator.choice(normal, len(normal)), generator.choice(anomalous, len(anomalous))])
            for name, v in values.items():
                result[name][i] += roc_auc_score(y[idx], v[idx])
    return {name: total / len(machines) for name, total in result.items()}


def summarise(samples: dict[str, np.ndarray]) -> list[str]:
    names = list(samples)
    lines = [f"{'run':36s} mean   95% interval"]
    for name in names:
        lo, hi = np.percentile(samples[name], [2.5, 97.5])
        lines.append(f"{name:36s} {samples[name].mean():.3f}  [{lo:.3f}, {hi:.3f}]")
    reference = names[0]
    if len(names) > 1:
        lines += ["", f"difference against {reference}:", f"{'run':36s} diff    95% interval        P(diff > 0)"]
        for name in names[1:]:
            diff = samples[name] - samples[reference]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            lines.append(f"{name:36s} {diff.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  {np.mean(diff > 0):.2f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", nargs="+", required=True, help="name=folder; the first one is the reference")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dev", action="store_true")
    mode.add_argument("--eval", action="store_true")
    parser.add_argument("--domain", default="source", choices=("source", "target"))
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    runs = {}
    for item in args.scores:
        name, sep, folder = item.partition("=")
        runs[name if sep else Path(item).name] = read_machine_scores(Path(folder if sep else item).expanduser())
    first = next(iter(runs.values()))
    if args.eval:
        labels = {m: eval_labels(m, domain=args.domain) for m in eval_machines()}
    else:
        labels = {m: dev_labels(first[m], args.domain) for m in first}
    print("\n".join(summarise(bootstrap_means(runs, labels, args.n))))


if __name__ == "__main__":
    main()
