"""Run the official DCASE 2026 Task 2 evaluator on score folders, and compare its score with the reconstruction of this repo.

    # evaluation set: the ground truth of the five evaluation machines ships with the evaluator
    python paper_benchmark/official_evaluate.py --scores a=<run>/teams/moviad/patchcore b=<run2>/teams/moviad/patchcore
    # dev machines: the ground truth is the one the runner wrote next to the scores of a run
    python paper_benchmark/official_evaluate.py --ground-truth <dev run>/ --scores a=...

For every score folder a workspace is built (ground truth copied, the scores copied under ``teams/moviad/<name>``, a
``decision_result`` file made with the 90th percentile of the scores of each machine, as the evaluator needs one), the real
``dcase2026_task2_evaluator.py`` is run on it, and its "official score" is read from ``teams_result/<name>_result.csv``. The score
that ``score_eval.py`` / ``run_eval.py`` reconstruct (harmonic mean of source AUC, target AUC and pAUC of every machine) is printed next to it.
The decisions only matter for the precision / recall / F1 columns, not for the official score.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import hmean

from fuse_scores import read_machine_scores
from score_eval import GT_ROOT, evaluate

EVALUATOR_DIR = Path(__file__).resolve().parent.parent / "dcase2026_task2_evaluator"
GROUND_TRUTH_DIRS = ("ground_truth_data", "ground_truth_domain", "ground_truth_attributes")


def reconstructed_score(per_machine: dict) -> float:
    """The score as ``run_eval.py`` computes it (same definition as the evaluator), x 100."""
    values = ([m["auc_source_official"] for m in per_machine.values()] + [m["auc_target_official"] for m in per_machine.values()]
              + [m["pauc_all"] for m in per_machine.values()])
    return float(hmean(np.maximum(np.array(values, dtype=float), 1e-12))) * 100


def write_decisions(scores_path: Path, decisions_path: Path, percentile: float = 90.0) -> None:
    with scores_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]
    threshold = np.percentile([float(row[1]) for row in rows], percentile)
    with decisions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows([[row[0], int(float(row[1]) > threshold)] for row in rows])


def build_workspace(workspace: Path, ground_truth: Path, systems: dict[str, Path]) -> None:
    for name in GROUND_TRUTH_DIRS:
        source = ground_truth / name
        if not source.is_dir():
            raise FileNotFoundError(f"{source} is missing: the evaluator needs {GROUND_TRUTH_DIRS}")
        shutil.copytree(source, workspace / name)
    for name, folder in systems.items():
        target = workspace / "teams" / "moviad" / name
        target.mkdir(parents=True)
        files = sorted(Path(folder).glob("anomaly_score_*_test.csv"))
        if not files:
            raise FileNotFoundError(f"No anomaly_score_*_test.csv in {folder}")
        for path in files:
            shutil.copy(path, target / path.name)
            write_decisions(path, target / path.name.replace("anomaly_score_", "decision_result_", 1))


def read_official_score(result_csv: Path) -> float:
    """The "official score" row of an evaluator result file, x 100."""
    with result_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row and row[0] == "official score":
                return float(row[-1]) * 100
    raise ValueError(f"No 'official score' row in {result_csv}")


def run_evaluator(systems: dict[str, Path], ground_truth: Path = GT_ROOT, keep: Path | None = None) -> dict[str, tuple[float, float]]:
    """``{name: (official score of the evaluator, reconstructed score)}``."""
    with tempfile.TemporaryDirectory() as temporary:
        workspace = keep or Path(temporary)
        build_workspace(workspace, ground_truth, systems)
        command = [sys.executable, "dcase2026_task2_evaluator.py", "--ground_truth_root", str(workspace), "--teams_root_dir", "./teams",
                   "--result_dir", "./teams_result", "--additional_result_dir", "./teams_additional_result", "--dir_depth", "2"]
        result = subprocess.run(command, cwd=EVALUATOR_DIR, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"The evaluator failed:\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
        scores = {}
        for name, folder in systems.items():
            official = read_official_score(workspace / "teams_result" / f"{name}_result.csv")
            scores[name] = (official, reconstructed_score(evaluate(read_machine_scores(Path(folder)), workspace)))
        return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", nargs="+", required=True, help="name=folder with anomaly_score_*_test.csv")
    parser.add_argument("--ground-truth", type=Path, default=GT_ROOT,
                        help="folder with ground_truth_data / _domain / _attributes (default: the evaluation ground truth of the evaluator)")
    args = parser.parse_args()
    systems = {}
    for item in args.scores:
        name, sep, folder = item.partition("=")
        systems[name if sep else Path(item).name] = Path(folder if sep else item).expanduser()
    print(f"{'system':44s} official (evaluator)  reconstructed  difference")
    for name, (official, mine) in run_evaluator(systems, args.ground_truth).items():
        print(f"{name:44s} {official:18.2f}  {mine:13.2f}  {official - mine:+.3f}")


if __name__ == "__main__":
    main()
