import numpy as np

from paper_benchmark.fuse_scores import fuse, machine_auc, normalise


def clips():
    return [f"section_00_source_test_{'anomaly' if i % 2 else 'normal'}_{i:04d}.wav" for i in range(20)]


def test_rank_ignores_scale_and_fusion_beats_a_weaker_member():
    rng = np.random.default_rng(0)
    files = clips()
    labels = np.array([i % 2 for i in range(20)])
    good = labels + rng.normal(0, 0.4, 20)
    noisy = labels + rng.normal(0, 0.4, 20)
    a = {"m": dict(zip(files, good * 1000))}
    b = {"m": dict(zip(files, noisy))}
    fused, skipped = fuse({"a": a, "b": b})
    assert not skipped
    assert machine_auc(fused["m"])[0] >= min(machine_auc(a["m"])[0], machine_auc(b["m"])[0])


def test_ties_share_rank_and_missing_machines_are_skipped():
    assert list(normalise(np.array([1.0, 1.0, 3.0]), "rank")) == [0.25, 0.25, 1.0]
    files = clips()
    one = {"m": dict.fromkeys(files, 1.0), "n": dict.fromkeys(files, 1.0)}
    two = {"m": dict.fromkeys(files, 2.0)}
    fused, skipped = fuse({"one": one, "two": two})
    assert list(fused) == ["m"] and skipped == {"n": ["two"]}
