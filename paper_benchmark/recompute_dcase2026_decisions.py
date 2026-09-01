"""Regenerate DCASE decision CSVs from saved train/test anomaly scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_pairs(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return [row[0] for row in rows], np.asarray([float(row[1]) for row in rows], dtype=float)


def _write_pairs(path: Path, names: list[str], values: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for name, value in sorted(zip(names, values)):
            writer.writerow([Path(name).name, value])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams-root", type=Path, default=Path("dcase2026_task2_evaluator/dev/teams/moviad"))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--percentile", type=float, default=99.0)
    args = parser.parse_args()
    if not 0.0 < args.percentile < 100.0:
        raise ValueError("--percentile must be between 0 and 100")

    methods = [p for p in sorted(args.teams_root.iterdir()) if p.is_dir()]
    if args.methods:
        selected = {method.lower() for method in args.methods}
        methods = [p for p in methods if p.name.lower() in selected]

    for method_dir in methods:
        train_files = sorted(method_dir.glob("train_anomaly_score_*_section_00_train.csv"))
        for train_path in train_files:
            stem = train_path.name.removeprefix("train_anomaly_score_").removesuffix("_section_00_train.csv")
            test_path = method_dir / f"anomaly_score_{stem}_section_00_test.csv"
            if not test_path.exists():
                print(f"Skipping {method_dir.name}/{stem}: test scores are missing")
                continue
            _, train_scores = _read_pairs(train_path)
            test_names, test_scores = _read_pairs(test_path)
            threshold = float(np.percentile(train_scores, args.percentile))
            decisions = (test_scores >= threshold).astype(int)
            decision_path = method_dir / f"decision_result_{stem}_section_00_test.csv"
            _write_pairs(decision_path, test_names, decisions)
            print(f"{method_dir.name}/{stem}: percentile={args.percentile:g}, threshold={threshold:.8g}")


if __name__ == "__main__":
    main()
