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
