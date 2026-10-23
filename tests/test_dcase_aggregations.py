import argparse
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchaudio

from paper_benchmark import benchmark_common as common
from paper_benchmark import run_dcase2026_task2 as runner


@pytest.mark.parametrize("name", ["max", "mean", "temporal_topk_mean", "top5pct", "top10pct", "top12.5pct", "top100pct"])
def test_valid_aggregation_names(name):
    assert runner.validate_aggregation(name) == name


@pytest.mark.parametrize("name", ["median", "top0pct", "top101pct", "top5", "topXpct", "pct5"])
def test_invalid_aggregation_names_are_rejected(name):
    with pytest.raises(argparse.ArgumentTypeError, match="Unsupported"):
        runner.validate_aggregation(name)


def test_top_percent_sits_between_max_and_mean():
    # one 4 x 5 map with the values 0..19
    anomaly_map = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
    output = (anomaly_map, torch.tensor([19.0]), torch.zeros(1, 4))

    def score(aggregation):
        return float(runner._aggregate_dcase_score(output, aggregation)[0])

    assert score("top5pct") == 19.0  # 5% of 20 values = 1 value: the maximum
    assert score("top50pct") == pytest.approx(14.5)  # mean of 10..19
    assert score("top100pct") == pytest.approx(score("mean")) == pytest.approx(9.5)
    assert score("top10pct") == pytest.approx(18.5)  # the 2 highest values


def test_top_percent_is_computed_per_clip():
    first = torch.arange(20, dtype=torch.float32)
    second = first.flip(0) * 2
    output = (torch.stack([first, second]).reshape(2, 1, 4, 5), torch.zeros(2), torch.zeros(2, 4))

    scores = runner._aggregate_dcase_score(output, "top10pct")

    np.testing.assert_allclose(scores, [18.5, 37.0])


def test_every_aggregation_comes_from_a_single_forward_pass():
    calls = []

    class FakeModel(torch.nn.Module):
        def forward(self, waveform):
            calls.append(1)
            maps = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5).repeat(waveform.shape[0], 1, 1, 1)
            return maps, maps.flatten(1).amax(1), torch.zeros(waveform.shape[0], 4)

    loader = [(torch.zeros(2, 8), torch.tensor([0, 1]), ("a.wav", "b.wav")), (torch.zeros(1, 8), torch.tensor([1]), ("c.wav",))]

    by_aggregation, labels, paths = runner._score_model_multi(
        FakeModel(), loader, "cpu", aggregations=("max", "mean", "top50pct")
    )

    assert len(calls) == 2  # two batches, not 2 batches x 3 aggregations
    assert set(by_aggregation) == {"max", "mean", "top50pct"} and all(len(v) == 3 for v in by_aggregation.values())
    np.testing.assert_allclose(by_aggregation["max"], 19.0)
    np.testing.assert_allclose(by_aggregation["top50pct"], 14.5)
    assert labels.tolist() == [0, 1, 1] and paths == ["a.wav", "b.wav", "c.wav"]
    single, _, _ = runner._score_model(FakeModel(), loader, "cpu", aggregation="mean")
    np.testing.assert_allclose(single, by_aggregation["mean"])


def write_clip(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    torchaudio.save(str(path), torch.randn(2, 32000, generator=generator) * 0.1, 16000)


def test_run_one_writes_the_extra_aggregations_next_to_the_main_one(tmp_path):
    dataset = tmp_path / "data"
    for index in range(6):
        write_clip(dataset / "fan" / "train" / f"section_00_source_train_normal_{index:04d}.wav", index)
    for index, label in enumerate(["normal", "normal", "anomaly", "anomaly"]):
        write_clip(dataset / "fan" / "test" / f"section_00_source_test_{label}_{index:04d}.wav", 100 + index)

    config = common.load_config(None)
    config.update(device="cpu", batch_size=2, pretrained=False, memory_bank_size=40, num_workers=0)
    config["dcase2026"] = {"train_domains": "source", "test_domain": "source", "extra_aggregations": ["max", "top5pct"]}
    args = SimpleNamespace(
        seed=0, debug=False, write_decisions=False, decision_percentile=99.0,
        score_aggregation=None, score_topk=None, extra_aggregations=None, epochs=None,
    )
    evaluator_root = tmp_path / "evaluator"

    row = runner.run_one("patchcore", config, dataset, "fan", 0, args, evaluator_root)

    teams = evaluator_root / "teams" / "moviad"
    for folder in ("patchcore", "patchcore@max", "patchcore@top5pct"):
        assert (teams / folder / "anomaly_score_fan_section_00_test.csv").exists()
        assert (teams / folder / "train_anomaly_score_fan_section_00_train.csv").exists()
    assert row["score_aggregation"] == "temporal_topk_mean"
    assert set(row["extra_aggregations"]) == {"max", "top5pct"}
    assert all("auc_source" in metrics for metrics in row["extra_aggregations"].values())
