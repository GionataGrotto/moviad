import csv
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_benchmark"))

import run_eval  # noqa: E402


def test_plan_pairs_exist_and_fusions_use_planned_pairs():
    from run_dcase_grid import EXPERIMENTS

    for experiment, method in run_eval.PAIRS:
        assert experiment in EXPERIMENTS
        assert EXPERIMENTS[experiment][2] is None or method in EXPERIMENTS[experiment][2]
    for parts, _ in run_eval.FUSIONS.values():
        assert set(parts) <= set(run_eval.PAIRS)
    assert run_eval.FIXED_FUSION in run_eval.FUSIONS
    for with_ls, without in run_eval.LS_PAIRS:
        assert with_ls in run_eval.DEV_AUC and without in run_eval.DEV_AUC
    assert len(run_eval.PAIRS) <= 20


def test_pair_runner_retries_and_uses_the_eval_suffix():
    calls = []

    def runner(command, cwd=None):
        calls.append(command)
        return types.SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    args = types.SimpleNamespace(base_config="c.json", epochs=30, retries=1, retry_wait=0)
    assert run_eval.run_pair(("baseline", "knn"), args, runner) is True and len(calls) == 2
    assert calls[0][calls[0].index("--run-suffix") + 1] == run_eval.SUFFIX
    assert "--no-summary" in calls[0]  # the dev summary cannot read the unlabelled evaluation clip names: it must not run for these pairs


def test_report_is_built_from_saved_scores(tmp_path):
    gt = tmp_path / "gt"
    for folder, values in (("ground_truth_data", [0, 1, 0, 1]), ("ground_truth_domain", [0, 0, 1, 1])):
        (gt / folder).mkdir(parents=True)
        with (gt / folder / "ground_truth_Toy_section_00_test.csv").open("w", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows([[f"section_00_{i:04d}.wav", v] for i, v in enumerate(values, 1)])
    for pair in run_eval.PAIRS:
        folder = run_eval.run_folder(pair, tmp_path)
        folder.mkdir(parents=True)
        with (folder / "anomaly_score_Toy_section_00_test.csv").open("w", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows([[f"section_00_{i:04d}.wav", s] for i, s in enumerate([0.1, 0.9, 0.2, 0.8], 1)])
    text = "\n".join(run_eval.build_report(tmp_path, gt))
    assert "fissata in anticipo" in text and "1.000" in text and "Verifica del LS" in text


def test_target_pairs_are_in_the_plan_and_the_official_score_is_a_harmonic_mean():
    for with_target, without in run_eval.TARGET_PAIRS:
        assert with_target in run_eval.DEV_AUC and without in run_eval.DEV_AUC
    result = {"a": {"auc_source_official": 0.8, "auc_target_official": 0.8, "pauc_all": 0.8},
              "b": {"auc_source_official": 0.8, "auc_target_official": 0.8, "pauc_all": 0.8}}
    assert abs(run_eval.official_score(result) - 80.0) < 1e-6
    result["b"]["auc_target_official"] = 0.4
    assert run_eval.official_score(result) < 80.0 * 0.95


def test_a_repeat_with_another_seed_is_run_with_its_own_suffix_and_seed():
    calls = []
    pair = ("tgt_pc_quota10_seed2", "patchcore")
    run_eval.SEEDED[pair] = ("tgt_pc_quota10", 2)

    def runner(command, cwd=None):
        calls.append(command)
        return types.SimpleNamespace(returncode=0)

    args = types.SimpleNamespace(base_config="c.json", epochs=30, retries=0, retry_wait=0)
    assert run_eval.run_pair(pair, args, runner) is True
    command = calls[0]
    assert command[command.index("--experiments") + 1] == "tgt_pc_quota10"
    assert command[command.index("--run-suffix") + 1] == "_seed2_eval" and command[command.index("--seed") + 1] == "2"
    assert run_eval.run_folder(pair, Path("/r")).parts[-4] == "grid_tgt_pc_quota10_seed2_eval"
    run_eval.SEEDED.pop(pair)


def test_fusion_specs_average_the_runs_joined_by_an_ampersand_and_report_missing_ones():
    import numpy as np

    files = [f"section_00_{i:04d}.wav" for i in range(1, 9)]
    rng = np.random.default_rng(0)
    scores = {(name, "m"): {"Toy": dict(zip(files, rng.normal(size=8)))} for name in ("a", "b", "b_seed1", "b_seed2")}
    assert run_eval.parse_fusion("a:m + b:m&b_seed1:m") == [[("a", "m")], [("b", "m"), ("b_seed1", "m")]]
    fused = run_eval.fuse_spec("a:m + b:m&b_seed1:m&b_seed2:m", scores)
    assert set(fused) == {"Toy"} and set(fused["Toy"]) == set(files)
    assert run_eval.fuse_spec("a:m + missing:m", scores) is None


def test_seed_groups_collect_runs_that_differ_only_by_the_seed():
    results = {("x", "patchcore"): 1, ("x_seed1", "patchcore"): 2, ("x_seed2", "patchcore"): 3, ("y", "patchcore"): 4, ("x", "knn"): 5}
    groups = dict(run_eval.seed_groups(results))
    assert list(groups) == ["x:patchcore"] and len(groups["x:patchcore"]) == 3


def test_every_eval_run_on_disk_is_listed_in_the_report(tmp_path):
    folder = tmp_path / "dcase2026_task2_evaluator" / "dev" / "grid_new_experiment_eval" / "teams" / "moviad"
    (folder / "patchcore").mkdir(parents=True)
    (folder / "patchcore@max").mkdir()
    run_eval.discover_pairs(tmp_path)
    assert ("new_experiment", "patchcore") in run_eval.PAIRS and ("new_experiment", "patchcore@max") not in run_eval.PAIRS
    run_eval.PAIRS.remove(("new_experiment", "patchcore"))
    run_eval.DEV_AUC.pop(("new_experiment", "patchcore"))
