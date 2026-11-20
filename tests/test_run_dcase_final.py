import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_benchmark"))

import run_dcase_final as final  # noqa: E402
import run_eval  # noqa: E402
from run_dcase_grid import EXPERIMENTS  # noqa: E402


def test_every_planned_experiment_exists_and_applies_to_its_method():
    for dinomaly in ("mask_tw", "tw"):
        for part, rows in final.plan(dinomaly).items():
            for label, experiment, method in rows:
                assert experiment in EXPERIMENTS, (part, experiment)
                allowed = EXPERIMENTS[experiment][2]
                assert allowed is None or method in allowed, (experiment, method)


def test_the_runs_are_distinct_and_seed_zero_keeps_the_name_of_run_eval():
    runs = final.run_pairs(("models", "ladder", "leave"), [0, 1, 2])
    assert len(runs) == len(set(runs))
    assert ("tgt_pc_quota10", "patchcore", 1) in runs
    assert final.pair_of("tgt_pc_quota10", "patchcore", 0) == ("tgt_pc_quota10", "patchcore")
    pair = final.pair_of("tgt_pc_quota10", "patchcore", 2)
    assert pair == ("tgt_pc_quota10_seed2", "patchcore") and run_eval.SEEDED[pair] == ("tgt_pc_quota10", 2)


def test_the_ablation_variants_remove_exactly_one_piece():
    best, no_ls, no_std, no_banks = (EXPERIMENTS[n][1] for n in ("tgt_pc_quota10", "abl_q10_no_ls", "abl_q10_no_std", "abl_q10_no_banks_std"))
    assert best["patchcore_band_banks"] and best["patchcore_score_std"] == "band" and best["dcase2026"]["channel"] == "ls"
    assert no_ls["dcase2026"]["channel"] == 0 and no_ls["patchcore_target_fraction"] == 0.10 and no_ls["patchcore_band_banks"]
    assert no_ls["patchcore_score_std"] == "band"
    assert "patchcore_score_std" not in no_std and no_std["patchcore_band_banks"] and no_std["dcase2026"]["channel"] == "ls"
    assert "patchcore_band_banks" not in no_banks and "patchcore_score_std" not in no_banks and no_banks["patchcore_target_fraction"] == 0.10


def test_the_report_shows_mean_std_and_the_changes():
    def run(official):
        return {"source": official + 10, "target": official - 5, "pauc": 55.0, "official": official}

    metrics = {}
    values = [(50, 51), (52, 53), (54, 55), (56, 57), (58, 59), (60, 61), (64, 65)]
    for (label, experiment, method), pair in zip(final.plan()["ladder"], values):
        for seed, value in enumerate(pair):
            metrics[(experiment, method, seed)] = run(value)
    text = "\n".join(final.build_report(("ladder",), [0, 1], metrics=metrics))
    assert "50.5 ± 0.7" in text and "+2.0" in text and "64.5 ± 0.7" in text
    assert final.cell([], "official") == "n/d"
    leave = final.table([("a", [run(64.0)]), ("b", [run(60.0)])], reference=0)
    assert leave[-1].endswith("-4.0 |") and leave[-2].endswith("|  |")
