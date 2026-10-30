import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_benchmark"))

from bootstrap_ci import bootstrap_means, dev_labels, summarise  # noqa: E402


def make_run(rng, signal):
    names = [f"section_00_source_test_{'anomaly' if i % 2 else 'normal'}_{i:04d}.wav" for i in range(60)]
    return {n: float((i % 2) * signal + rng.normal(0, 1)) for i, n in enumerate(names)}


def test_interval_contains_the_estimate_and_a_better_run_wins_the_paired_difference():
    rng = np.random.default_rng(0)
    weak, strong = make_run(rng, 0.2), make_run(rng, 2.5)
    labels = {"m": dev_labels(weak)}
    samples = bootstrap_means({"weak": {"m": weak}, "strong": {"m": strong}}, labels, n=200, seed=1)
    lo, hi = np.percentile(samples["strong"], [2.5, 97.5])
    assert lo < samples["strong"].mean() < hi
    assert (samples["strong"] - samples["weak"]).mean() > 0.2
    text = "\n".join(summarise(samples))
    assert "difference against weak" in text and "1.00" in text


def test_only_source_clips_are_used():
    scores = {"section_00_source_test_normal_0000.wav": 0.1, "section_00_target_test_anomaly_0001.wav": 0.9}
    assert dev_labels(scores) == {"section_00_source_test_normal_0000.wav": 0}
