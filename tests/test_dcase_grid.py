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


def test_beats_search_experiments_change_one_factor_around_beats_late():
    base = {"dcase2026": {"dataset_path": "/data", "channel": 0}}
    late = grid.build_config(base, "beats_late", 30)
    assert late["backbone"] == "BEATs" and late["layers"] == [8, 10, 12]
    assert late["dcase2026"]["train_domains"] == "source" and late["dcase2026"]["channel"] == 0

    assert grid.build_config(base, "beats_b12", 30)["layers"] == [12]
    assert grid.build_config(base, "beats_b10_12", 30)["layers"] == [10, 11, 12]
    train_all = grid.build_config(base, "beats_late_train_all", 30)
    assert train_all["layers"] == [8, 10, 12] and train_all["dcase2026"]["train_domains"] == "all"
    assert grid.build_config(base, "beats_late_diff", 30)["dcase2026"]["channel"] == "diff"
    assert grid.build_config(base, "beats_late_agg_mean", 30)["dcase2026"]["score_aggregation"] == "mean"
    assert grid.build_config(base, "beats_padim_full_cov", 30)["padim_diag_cov"] is False
    assert grid.build_config(base, "beats_padim_d576", 30)["padim_embedding_dim"] == 576
    assert grid.build_config(base, "beats_pc_bank60k", 30)["memory_bank_size"] == 60000
    assert grid.build_config(base, "beats_cfa_gamma1", 30)["cfa_gamma_c"] == 1
    for name in ("beats_padim_full_cov", "beats_padim_d576"):
        assert grid.methods_for(name, ["patchcore", "padim", "cfa"]) == ["padim"]
    assert grid.methods_for("beats_b12", ["patchcore", "padim", "cfa", "stfpm", "dinomaly"]) == ["patchcore", "padim", "cfa"]


def test_presets_only_name_existing_experiments_and_cover_the_search():
    for names in grid.PRESETS.values():
        assert names and all(name in grid.EXPERIMENTS for name in names) and len(set(names)) == len(names)
    assert set(grid.PRESETS["beats_search"]) == set(grid.PRESETS["beats_layers"]) | set(grid.PRESETS["beats_protocol"]) | set(grid.PRESETS["beats_heads"])


def test_preset_expands_to_its_experiments_and_cannot_be_mixed_with_a_list(tmp_path, monkeypatch, capsys):
    config = tmp_path / "base.json"
    config.write_text(json.dumps({"dcase2026": {"dataset_path": str(tmp_path)}}))
    argv = ["run_dcase_grid.py", "--base-config", str(config), "--methods", "padim", "--dry-run"]

    monkeypatch.setattr("sys.argv", argv + ["--preset", "beats_heads"])
    grid.main()
    output = capsys.readouterr().out
    assert "=== beats_padim_full_cov" in output and "=== beats_padim_d576" in output
    assert "beats_pc_bank60k" not in output  # PatchCore-only experiment: nothing to run for padim

    monkeypatch.setattr("sys.argv", argv + ["--preset", "beats_heads", "--experiments", "baseline"])
    with pytest.raises(SystemExit):
        grid.main()


def test_best_per_method_ignores_incomplete_runs():
    table = {
        ("a", "padim"): (0.52, 7), ("b", "padim"): (0.60, 5), ("c", "padim"): (0.55, 7),
        ("a", "cfa"): None, ("b", "cfa"): (0.58, 7), ("c", "cfa"): (0.57, 7),
    }
    best = grid.best_per_method(table, ["a", "b", "c"], ["padim", "cfa", "patchcore"])
    assert best == {"padim": ("c", 0.55), "cfa": ("b", 0.58)}  # the 5-machine 0.60 does not count


def test_skip_done_flag_drops_complete_pairs_from_the_commands(tmp_path, monkeypatch, capsys):
    config = tmp_path / "base.json"
    config.write_text(json.dumps({"dcase2026": {"dataset_path": str(tmp_path)}}))
    monkeypatch.setattr(grid, "pending_methods", lambda name, methods, suffix="", root=grid.ROOT: [m for m in methods if m != "padim"])
    monkeypatch.setattr("sys.argv", ["run_dcase_grid.py", "--base-config", str(config), "--methods", "patchcore", "padim", "cfa",
                                     "--experiments", "beats_late", "--skip-done", "--dry-run"])

    grid.main()

    output = capsys.readouterr().out
    assert "--methods patchcore cfa" in output and "padim" not in output.split("$")[1]
    assert "already complete for ['padim']" in output


def test_band_experiments_build_on_beats_late_and_change_one_thing():
    base = {"dcase2026": {"dataset_path": "/data"}}
    for name in ("band_pc_banks", "band_pc_std", "band_pc_banks_std", "band_padim_gauss",
                 "band_padim_gauss_full", "band_padim_gauss_full_std"):
        config = grid.build_config(base, name, 30)
        assert config["backbone"] == "BEATs" and config["layers"] == [8, 10, 12]
    assert grid.build_config(base, "band_pc_banks", 30)["patchcore_band_banks"] is True
    assert grid.build_config(base, "band_pc_std", 30)["patchcore_score_std"] == "band"
    both = grid.build_config(base, "band_pc_banks_std", 30)
    assert both["patchcore_band_banks"] is True and both["patchcore_score_std"] == "band"
    assert grid.build_config(base, "band_padim_gauss", 30)["padim_per_band"] is True
    full = grid.build_config(base, "band_padim_gauss_full", 30)
    assert full["padim_per_band"] is True and full["padim_diag_cov"] is False
    assert grid.build_config(base, "band_padim_gauss_full_std", 30)["padim_score_std"] == "band"
    assert grid.build_config(base, "dinomaly_l2", 30)["dinomaly_distance"] == "l2"
    assert grid.methods_for("band_pc_std", ["patchcore", "padim", "cfa"]) == ["patchcore"]
    assert grid.methods_for("band_padim_gauss_full", ["patchcore", "padim", "cfa"]) == ["padim"]
    assert grid.methods_for("dinomaly_l2", ["patchcore", "dinomaly"]) == ["dinomaly"]
    assert grid.PRESETS["band_search"] == [
        "band_pc_banks", "band_pc_std", "band_pc_banks_std", "band_padim_gauss", "band_padim_gauss_full", "band_padim_gauss_full_std",
    ]


def test_extra_aggregations_reach_the_config_and_the_summary_columns():
    base = {"dcase2026": {"dataset_path": "/data"}}
    config = grid.build_config(base, "baseline", 30, extra_aggregations=("max", "mean", "temporal_topk_mean"))
    assert config["dcase2026"]["extra_aggregations"] == ["max", "mean"]  # the primary one is not repeated
    assert "extra_aggregations" not in grid.build_config(base, "baseline", 30)["dcase2026"]

    assert grid.summary_labels(["patchcore", "padim"], ("max", "top5pct")) == [
        "patchcore", "padim", "patchcore@max", "patchcore@top5pct", "padim@max", "padim@top5pct",
    ]
    assert grid.summary_labels(["cfa"]) == ["cfa"]


def test_aggregation_sweep_flag_writes_the_extra_aggregations_into_the_configs(tmp_path, monkeypatch, capsys):
    config = tmp_path / "base.json"
    config.write_text(json.dumps({"dcase2026": {"dataset_path": str(tmp_path)}}))
    monkeypatch.setattr("sys.argv", ["run_dcase_grid.py", "--base-config", str(config), "--methods", "patchcore",
                                     "--experiments", "baseline", "--aggregation-sweep", "--dry-run"])

    grid.main()

    written = json.loads((grid.ROOT / "paper_benchmark" / "grid_configs" / "config.grid_baseline.json").read_text())
    assert written["dcase2026"]["extra_aggregations"] == ["max", "mean", "top5pct", "top10pct"]


def test_stfpm_experiments_set_the_optimizer_and_only_run_stfpm():
    base = {"dcase2026": {"dataset_path": "/data"}}
    assert grid.build_config(base, "stfpm_sgd_lr0p1", 30)["stfpm_lr"] == 0.1
    assert grid.build_config(base, "stfpm_sgd_lr0p02", 30)["stfpm_lr"] == 0.02
    adamw = grid.build_config(base, "stfpm_adamw_1e4", 30)
    assert adamw["stfpm_optimizer"] == "adamw" and adamw["stfpm_lr"] == 1e-4 and adamw["backbone"] == "Cnn14"
    late = grid.build_config(base, "stfpm_beats_late_adamw_1e5", 30)
    assert late["backbone"] == "BEATs" and late["layers"] == [8, 10, 12] and late["stfpm_lr"] == 1e-5
    assert grid.build_config(base, "stfpm_beats_b12_adamw_1e5", 30)["layers"] == [12]
    for name in ("stfpm_sgd_lr0p1", "stfpm_adamw_1e4", "stfpm_beats_late_adamw_1e4"):
        assert grid.methods_for(name, ["patchcore", "stfpm", "cfa"]) == ["stfpm"]
    assert "baseline" in grid.PRESETS["stfpm_search"] and "layers_deep" in grid.PRESETS["stfpm_search"]
