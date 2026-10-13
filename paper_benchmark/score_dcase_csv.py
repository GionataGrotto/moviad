"""AUC / pAUC per machine from DCASE 2026 Task 2 ``anomaly_score_*_test.csv`` files.

Works on the output folder of any system (this repo, or an external one) as long as the
file names keep the DCASE convention ``section_00_<source|target>_test_<normal|anomaly>_NNNN...``.

    python paper_benchmark/score_dcase_csv.py ~/Downloads/knn ~/Downloads/padim
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

SCORE_FILE = re.compile(r"^anomaly_score_(?P<machine>.+)_section_(?P<section>\d+)_test\.csv$")


def read_scores(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for name, score, *_ in csv.reader(handle):
            domain = "source" if "_source_" in name else "target" if "_target_" in name else None
            label = 1 if "_anomaly_" in name else 0 if "_normal_" in name else None
            if domain is None or label is None:
                raise ValueError(f"Cannot read domain/label from file name {name!r} in {path}")
            rows.append((domain, label, float(score)))
    return rows


def auc_pair(labels, scores):
    if len(set(labels)) < 2:
        return None, None
    return roc_auc_score(labels, scores), roc_auc_score(labels, scores, max_fpr=0.1)


def evaluate_folder(folder: Path, domain: str):
    results = {}
    for path in sorted(folder.glob("anomaly_score_*_test.csv")):
        match = SCORE_FILE.match(path.name)
        if match is None:
            continue
        rows = [r for r in read_scores(path) if domain == "all" or r[0] == domain]
        results[match["machine"]] = auc_pair([r[1] for r in rows], [r[2] for r in rows]) + (len(rows),)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", type=Path, help="folders with anomaly_score_*_test.csv files")
    parser.add_argument("--domain", choices=("source", "target", "all"), default="source")
    args = parser.parse_args()

    for folder in args.folders:
        results = evaluate_folder(folder.expanduser(), args.domain)
        if not results:
            print(f"== {folder}: no anomaly_score_*_test.csv files found", file=sys.stderr)
            continue
        print(f"== {folder}  (domain: {args.domain})")
        for machine, (auc, pauc, count) in results.items():
            if auc is None:
                print(f"{machine:24s} n={count:3d}  only one class, AUC undefined")
            else:
                print(f"{machine:24s} n={count:3d}  AUC={auc:.4f}  pAUC={pauc:.4f}")
        valid = [(a, p) for a, p, _ in results.values() if a is not None]
        if valid:
            print(f"{'MEAN':24s}         AUC={sum(a for a, _ in valid) / len(valid):.4f}  "
                  f"pAUC={sum(p for _, p in valid) / len(valid):.4f}\n")


if __name__ == "__main__":
    main()
