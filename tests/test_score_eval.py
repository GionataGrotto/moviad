import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_benchmark"))

from score_eval import eval_machines, evaluate, machine_metrics, mean_metric  # noqa: E402


def make_gt(root: Path):
    names = [f"section_00_{i:04d}.wav" for i in range(1, 9)]
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    domains = [0, 0, 0, 0, 1, 1, 1, 1]
    for folder, values in (("ground_truth_data", labels), ("ground_truth_domain", domains)):
        (root / folder).mkdir(parents=True)
        (root / folder / "ground_truth_Toy_section_00_test.csv").write_text("".join(f"{n},{v}\n" for n, v in zip(names, values)))
    return names


def test_source_and_target_auc_use_the_domain_file(tmp_path):
    names = make_gt(tmp_path)
    # perfect on the source clips, reversed on the target ones
    scores = dict(zip(names, [0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.8, 0.2]))
    metrics = evaluate({"Toy": scores}, tmp_path)["Toy"]
    assert metrics["auc_source"] == 1.0 and metrics["auc_target"] == 0.0
    assert eval_machines(tmp_path) == ["Toy"]
    assert mean_metric({"Toy": metrics}) == 1.0


def test_missing_scores_are_an_error(tmp_path):
    names = make_gt(tmp_path)
    with pytest.raises(ValueError, match="no score"):
        machine_metrics({names[0]: 0.1}, {n: 0 for n in names}, {n: 0 for n in names})
