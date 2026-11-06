import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_benchmark"))

import official_evaluate as official  # noqa: E402
from score_eval import evaluate  # noqa: E402

MACHINES = ("ToyA", "ToyB")


def make_ground_truth(root: Path):
    """Two machines, 16 test clips each: source and target, normal and anomalous (4 of every kind), named like the evaluation clips."""
    names = [f"section_00_{i:04d}.wav" for i in range(1, 17)]
    labels = [0, 0, 0, 0, 1, 1, 1, 1] * 2  # clips 1-8 source, 9-16 target
    domains = [0] * 8 + [1] * 8
    for folder, values in (("ground_truth_data", labels), ("ground_truth_domain", domains)):
        (root / folder).mkdir(parents=True)
        for machine in MACHINES:
            with (root / folder / f"ground_truth_{machine}_section_00_test.csv").open("w", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerows(zip(names, values))
    (root / "ground_truth_attributes").mkdir()
    for machine in MACHINES:
        with (root / "ground_truth_attributes" / f"ground_truth_{machine}_section_00_test.csv").open("w", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows([[f"section_00_target_test_normal_{i:04d}", f"section_00_target_test_normal_{i:04d}"] for i in range(1, 17)])
    return names, np.array(labels), np.array(domains)


def test_the_auc_of_a_domain_keeps_all_the_anomalous_clips(tmp_path):
    names, labels, domains = make_ground_truth(tmp_path / "gt")
    # source anomalies score high, target anomalies score low (below the source normals): the domain-only AUC of the source is perfect,
    # the evaluator's one (source normals against ALL the anomalies) is not
    rng = np.random.default_rng(0)
    scores = np.where(labels == 1, np.where(domains == 0, 0.9, 0.1), 0.5) + 0.01 * rng.normal(size=16)
    folder = tmp_path / "scores"
    folder.mkdir()
    for machine in MACHINES:
        with (folder / f"anomaly_score_{machine}_section_00_test.csv").open("w", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows(zip(names, scores))
    result = evaluate(official.read_machine_scores(folder), tmp_path / "gt")["ToyA"]
    assert result["auc_source"] == pytest.approx(1.0)
    assert result["auc_source_official"] == pytest.approx(0.5, abs=0.2) and result["auc_source_official"] < result["auc_source"]


def test_the_reconstruction_equals_the_official_evaluator(tmp_path):
    names, labels, domains = make_ground_truth(tmp_path / "gt")
    rng = np.random.default_rng(1)
    folder = tmp_path / "scores"
    folder.mkdir()
    for machine in MACHINES:
        scores = labels + 0.8 * rng.normal(size=16)
        with (folder / f"anomaly_score_{machine}_section_00_test.csv").open("w", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows(zip(names, scores))
    outcome = official.run_evaluator({"system": folder}, tmp_path / "gt")["system"]
    assert outcome[0] == pytest.approx(outcome[1], abs=1e-6)  # the evaluator's score and the reconstructed one are the same number
