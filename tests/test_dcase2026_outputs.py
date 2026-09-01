from pathlib import Path

import numpy as np
import torch

from paper_benchmark.run_dcase2026_task2 import (
    _aggregate_dcase_score,
    _write_pairs,
)


def test_dcase_output_is_headerless_and_sorted(tmp_path):
    output = tmp_path / "scores.csv"
    _write_pairs(output, ["b.wav", "a.wav"], [2.0, 1.0])
    assert output.read_text(encoding="utf-8") == "a.wav,1.0\nb.wav,2.0\n"


def test_dcase_temporal_topk_aggregation():
    anomaly_map = torch.tensor(
        [[[[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]]],
         [[[2.0, 1.0, 0.0], [4.0, 3.0, 6.0]]]]
    )
    scores = _aggregate_dcase_score(
        (anomaly_map, torch.tensor([99.0, 99.0])), "temporal_topk_mean", top_k=2
    )
    np.testing.assert_allclose(scores, [3.5, 3.25])


def test_dcase_max_aggregation_remains_available():
    scores = _aggregate_dcase_score(
        (torch.zeros(2, 1, 2, 2), torch.tensor([0.25, 0.75])), "max"
    )
    np.testing.assert_allclose(scores, [0.25, 0.75])
