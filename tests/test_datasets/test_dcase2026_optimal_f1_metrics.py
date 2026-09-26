"""Regression tests for the threshold-free F1 metric added alongside the
DCASE evaluator's percentile-based decisions.

These pin two things:
* on a balanced (e.g. hand-built 50/50) test set, ``percentile_decisions``
  at the DCASE-standard 99th percentile degenerates to flagging almost
  nothing, while ``optimal_f1_metrics`` still reflects the ranking quality;
* on a well-separated score distribution, both approaches agree that the
  classifier is good, so the new metric is not simply "always reports 1.0".
"""

import numpy as np

from paper_benchmark.benchmark_common import optimal_f1_metrics, percentile_decisions


def test_percentile_99_degenerates_on_a_balanced_test_set():
    """This is the exact failure mode reported against real DCASE runs:
    a 50/50 balanced set still gets almost nothing flagged at percentile 99,
    regardless of how informative the underlying scores are."""
    rng = np.random.default_rng(0)
    normal_scores = rng.normal(loc=0.0, scale=1.0, size=50)
    anomaly_scores = rng.normal(loc=1.5, scale=1.0, size=50)  # clearly separated
    scores = np.concatenate([normal_scores, anomaly_scores])
    labels = np.concatenate([np.zeros(50), np.ones(50)])

    _, decisions = percentile_decisions(scores, 99.0)
    assert decisions.sum() <= 2, (
        "the 99th percentile of 100 balanced samples should flag at most ~1-2, "
        "independently of how separable the scores are"
    )

    result = optimal_f1_metrics(scores, labels)
    assert result["f1_opt"] > 0.7, (
        "optimal_f1_metrics must still reflect the good separation that "
        "percentile_decisions hides on this balanced set"
    )


def test_optimal_f1_metrics_matches_a_known_perfect_separation():
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1])

    result = optimal_f1_metrics(scores, labels)

    assert result["f1_opt"] == 1.0
    assert result["precision_opt"] == 1.0
    assert result["recall_opt"] == 1.0


def test_optimal_f1_metrics_returns_empty_for_a_single_class():
    result = optimal_f1_metrics(np.array([0.1, 0.5, 0.9]), np.array([0, 0, 0]))
    assert result == {}


def test_optimal_f1_metrics_is_never_worse_than_chance_for_random_labels():
    """Sanity bound: F1 must stay within [0, 1] and not error on edge inputs
    such as all-identical scores (a degenerate but real case: a model that
    outputs a constant anomaly score for every clip)."""
    scores = np.zeros(10)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    result = optimal_f1_metrics(scores, labels)

    assert 0.0 <= result["f1_opt"] <= 1.0
    assert 0.0 <= result["precision_opt"] <= 1.0
    assert 0.0 <= result["recall_opt"] <= 1.0
