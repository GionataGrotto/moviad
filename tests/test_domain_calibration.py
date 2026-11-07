import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import domain_calibration as dc  # noqa: E402
from score_eval import machine_metrics  # noqa: E402
from test_official_evaluate import make_ground_truth  # noqa: E402


def test_removing_the_domain_offset_fixes_a_detector_that_ranks_well_inside_each_domain(tmp_path):
    names, labels, domains = make_ground_truth(tmp_path / "gt")
    rng = np.random.default_rng(0)
    # perfect inside each domain, but the target scores are shifted up by 5: every target normal is above every source anomaly
    scores = labels + 0.1 * rng.normal(size=16) + 5.0 * domains
    label_map, domain_map, score_map = dict(zip(names, labels)), dict(zip(names, domains)), dict(zip(names, scores))
    raw = machine_metrics(score_map, label_map, domain_map)
    fixed = machine_metrics(dc.calibrate(score_map, label_map, domain_map, "shift"), label_map, domain_map)
    assert raw["auc_target_official"] < 0.8 and fixed["auc_target_official"] == pytest.approx(1.0)
    assert dc.offset_in_std(score_map, label_map, domain_map) > 5
