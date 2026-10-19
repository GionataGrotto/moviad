import json

import numpy as np
import pytest
import torch
import torchaudio

from moviad.datasets.dcase2026_task2 import DCASE2026Task2Dataset
from moviad.models.audio.padim.padim import Padim
from paper_benchmark import run_dcase_grid as grid


def test_every_experiment_changes_one_factor_and_fixes_the_test_protocol():
    base = {"dcase2026": {"dataset_path": "/data", "channel": 0}}
    baseline = grid.build_config(base, "baseline", 5)

    assert baseline["dcase2026"]["train_domains"] == "source" and baseline["dcase2026"]["test_domain"] == "source"
    assert baseline["layers"] == ["conv_block2", "conv_block3", "conv_block4"] and baseline["epochs"] == 5
    assert grid.build_config(base, "layers_deep", 5)["layers"] == ["conv_block3", "conv_block4", "conv_block5"]
    assert grid.build_config(base, "train_all", 5)["dcase2026"]["train_domains"] == "all"
    assert grid.build_config(base, "near_minus_far", 5)["dcase2026"]["channel"] == "diff"
    for name in grid.EXPERIMENTS:
        config = grid.build_config(base, name, 5)
        assert config["dcase2026"]["test_domain"] == "source"
        assert config["dcase2026"]["run_name"] == f"grid_{name}"
        assert config["dcase2026"]["dataset_path"] == "/data"  # untouched keys survive the merge
    assert base["dcase2026"] == {"dataset_path": "/data", "channel": 0}  # the base config is not mutated


def test_method_specific_experiments_and_epoch_overrides():
    base = {"dcase2026": {"dataset_path": "/data"}}
    assert grid.build_config(base, "baseline", 30)["epochs"] == 30
    assert grid.build_config(base, "epochs_long", 30)["epochs"] == 100  # the experiment wins over --epochs
    assert grid.build_config(base, "stfpm_random_student", 30)["stfpm_student_pretrained"] is False
    assert grid.build_config(base, "patchcore_standardize", 30)["patchcore_standardize"] is True
    assert grid.build_config(base, "dinomaly_base", 30)["dinomaly_encoder"].startswith("deit_base")
    assert grid.build_config(base, "agg_mean", 30)["dcase2026"]["score_aggregation"] == "mean"  # the experiment wins
    assert grid.build_config(base, "baseline", 30)["dcase2026"]["score_aggregation"] == "temporal_topk_mean"
    assert grid.build_config(base, "spec_band_norm", 30)["spec_band_norm"] == "zscore"
    assert grid.build_config(base, "spec_fill_dead_bands", 30)["spec_dead_band_hz"] == 8000
    both = grid.build_config(base, "spec_both", 30)
    assert both["spec_band_norm"] == "zscore" and both["spec_dead_band_hz"] == 8000
    assert grid.methods_for("spec_both", ["patchcore", "dinomaly", "draem"]) == ["patchcore", "dinomaly"]
    assert grid.methods_for("agg_mean", ["patchcore", "dinomaly"]) == ["patchcore", "dinomaly"]
    beats = grid.build_config(base, "beats_late", 30)
    assert beats["backbone"] == "BEATs" and beats["layers"] == [8, 10, 12]
    assert grid.build_config(base, "beats_mid", 30)["layers"] == [4, 6, 8]
    assert grid.methods_for("beats_late", ["patchcore", "padim", "cfa", "stfpm", "dinomaly"]) == ["patchcore", "padim", "cfa"]
    assert grid.methods_for("stfpm_random_student", ["patchcore", "stfpm"]) == ["stfpm"]
    assert grid.methods_for("epochs_long", ["patchcore", "padim", "cfa", "stfpm", "dinomaly"]) == ["cfa", "stfpm", "dinomaly"]


def test_layer_experiments_skip_the_methods_that_have_no_layers():
    assert grid.methods_for("layers_fine", ["patchcore", "dinomaly", "cfa"]) == ["patchcore", "cfa"]
    assert grid.methods_for("baseline", ["patchcore", "dinomaly"]) == ["patchcore", "dinomaly"]


def test_padim_accepts_any_cnn14_block_combination():
    assert Padim.lookup_embedding_size("Cnn14", ("conv_block2", "conv_block3", "conv_block4")) == (896, 225)
    total, default = Padim.lookup_embedding_size("Cnn14", ("conv_block1", "conv_block2", "conv_block3"))
    assert total == 64 + 128 + 256 and 1 <= default <= total
    assert Padim.lookup_embedding_size("Cnn14", ("conv_block3", "conv_block4", "conv_block5"))[0] == 1792
    with pytest.raises(KeyError, match="Unsupported layer combination"):
        Padim.lookup_embedding_size("Cnn14", ("conv_block9",))


def write_clip(path, near, far):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), torch.stack([near, far]), 16000)


@pytest.mark.parametrize("channel", [0, 1, "diff"])
def test_dcase_dataset_channel_selection(tmp_path, channel):
    near = torch.linspace(-0.5, 0.5, 1600)
    far = torch.full((1600,), 0.25)
    write_clip(tmp_path / "fan" / "train" / "section_00_source_train_normal_0000.wav", near, far)

    dataset = DCASE2026Task2Dataset(tmp_path, "fan", "train", channel=channel, target_sample_rate=16000)
    waveform = dataset[0]

    expected = {0: near, 1: far, "diff": near - far}[channel]
    np.testing.assert_allclose(waveform.numpy(), expected.numpy(), atol=1e-3)


def test_dcase_dataset_rejects_a_bad_channel(tmp_path):
    with pytest.raises(ValueError, match="channel"):
        DCASE2026Task2Dataset(tmp_path, "fan", "train", channel=-1)


def test_dry_run_prints_one_command_per_experiment_without_running(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"dcase2026": {"dataset_path": str(tmp_path)}}))
    monkeypatch.setattr("sys.argv", ["run_dcase_grid.py", "--base-config", str(base), "--methods", "padim", "dinomaly",
                                     "--experiments", "baseline", "layers_deep", "--dry-run"])
    ran = []
    monkeypatch.setattr(grid.subprocess, "run", lambda *a, **k: ran.append(a))

    grid.main()

    output = capsys.readouterr().out
    assert not ran
    assert output.count("run_dcase2026_task2.py") == 2
    assert "--methods padim dinomaly" in output and "--methods padim\n" in output  # dinomaly skipped for layers_deep


def test_signal_feature_diagnostic_detects_a_louder_anomaly(tmp_path, capsys, monkeypatch):
    from paper_benchmark import diagnose_signal_features as diagnose

    torch.manual_seed(0)
    for index in range(6):
        for label, gain in (("normal", 0.1), ("anomaly", 0.4)):
            clip = torch.randn(16000) * gain
            write_clip(
                tmp_path / "fan" / "test" / f"section_00_source_test_{label}_{index:04d}.wav", clip, clip
            )
    monkeypatch.setattr("sys.argv", ["diagnose_signal_features.py", "--dataset-path", str(tmp_path)])

    diagnose.main()

    row = [line for line in capsys.readouterr().out.splitlines() if line.startswith("fan")][0].split()
    assert float(row[1]) == 1.0  # rms: every anomaly is louder than every normal clip


def test_seed_and_run_suffix_keep_repeated_runs_apart(tmp_path, monkeypatch, capsys):
    base = {"dcase2026": {"dataset_path": "/data"}}
    assert grid.build_config(base, "baseline", 5, suffix="_seed2")["dcase2026"]["run_name"] == "grid_baseline_seed2"
    assert grid.build_config(base, "baseline", 5)["dcase2026"]["run_name"] == "grid_baseline"

    config = tmp_path / "base.json"
    config.write_text(json.dumps(base))
    monkeypatch.setattr("sys.argv", ["run_dcase_grid.py", "--base-config", str(config), "--methods", "dinomaly",
                                     "--experiments", "baseline", "--seed", "7", "--run-suffix", "_seed7", "--dry-run"])
    grid.main()

    output = capsys.readouterr().out
    assert "--seed 7" in output
    written = json.loads((grid.ROOT / "paper_benchmark" / "grid_configs" / "config.grid_baseline_seed7.json").read_text())
    assert written["dcase2026"]["run_name"] == "grid_baseline_seed7"
