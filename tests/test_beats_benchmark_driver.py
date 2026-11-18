import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_benchmark"))

import build_benchmark_tables as tables  # noqa: E402
import run_beats_benchmark as driver  # noqa: E402
from benchmark_common import completed_runs, run_key  # noqa: E402

BASE = {"device": "auto", "output_dir": "results/paper_benchmark", "backbone": "Cnn14", "layers": ["conv_block2"], "epochs": 1,
        "mimii": {"dataset_path": "m", "seeds": [42, 43]}, "envmix": {"seeds": [0, 1, 2]}, "faithfulness": {"enabled": True}}


def test_every_variant_is_beats_in_its_own_folder_with_one_seed_and_no_faithfulness():
    for variant, (method, _) in driver.VARIANTS.items():
        config = driver.build_config(BASE, "mimii", variant, "results/x", [42], [0])
        assert config["backbone"] == "BEATs" and config["variant"] == variant and config["methods"] == [method]
        assert config["output_dir"] == "results/x" and config["faithfulness"] == {"enabled": False}
        assert config["mimii"]["seeds"] == [42] and config["envmix"]["seeds"] == [0] and config["mimii"]["dataset_path"] == "m"
    patchcore = driver.build_config(BASE, "envmix", "patchcore_beats_std", "o", [42], [0])
    assert patchcore["patchcore_band_banks"] and patchcore["patchcore_score_std"] == "band" and patchcore["layers"] == [8, 10, 12]
    assert BASE["backbone"] == "Cnn14"  # the base config is not modified


def test_the_plan_runs_the_cheap_methods_first_and_can_be_filtered():
    entries = driver.plan(["mimii", "envmix"], None)
    assert entries[0] == ("mimii", "patchcore_beats_std") and entries[-1] == ("envmix", "stfpm_beats") and len(entries) == 2 * len(driver.VARIANTS)
    assert driver.plan(["mimii"], ["padim"]) == [("mimii", "padim_beats")]
    assert driver.plan(["mimii"], ["dinomaly_beats"]) == [("mimii", "dinomaly_beats")]


def test_skip_done_reads_the_keys_of_the_rows_already_written(tmp_path):
    path = tmp_path / "r.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "method", "background_category", "snr_db", "seed"])
        writer.writeheader()
        writer.writerow({"variant": "v", "method": "patchcore", "background_category": "siren", "snr_db": 6.0, "seed": 0})
    keys = ("variant", "method", "background_category", "snr_db", "seed")
    done = completed_runs(path, keys)
    assert run_key({"variant": "v", "method": "patchcore", "background_category": "siren", "snr_db": float(6), "seed": int(0)}, keys) in done
    assert run_key({"variant": "v", "method": "padim", "background_category": "siren", "snr_db": 6.0, "seed": 0}, keys) not in done
    assert completed_runs(tmp_path / "missing.csv", keys) == set()


def test_the_tables_hold_the_paper_rows_and_ours_and_bold_the_best():
    mimii_new = [{"variant": "patchcore_beats_std", "method": "patchcore", "category": c, "img_roc_auc": 0.95, "f1_img": 0.9} for c in tables.CATEGORIES]
    text = tables.mimii_table(mimii_new)
    assert "PatchCore [paper]" in text and "[BEATs]" in text and "Cnn14" not in text and "rerun" not in text
    new_line = [line for line in text.splitlines() if "BEATs" in line][0]
    assert "**95.0**" in new_line and "**90.0**" in new_line  # best of its columns
    envmix_new = [{"variant": "padim_beats", "method": "padim", **{column: 0.5 for _, column in tables.ENVMIX_COLUMNS}}]
    env = tables.envmix_table(envmix_new)
    assert "PaDiM, BEATs [BEATs]" in env and "Cnn14" not in env


def test_debug_holds_out_every_other_batch_for_the_score_statistics():
    assert "score_std_holdout" not in driver.build_config(BASE, "mimii", "patchcore_beats_std", "o", [42], [0])
    assert driver.build_config(BASE, "mimii", "patchcore_beats_std", "o", [42], [0], debug=True)["score_std_holdout"] == 0.5


def test_the_runners_join_numeric_beats_layers_into_the_csv_row():
    for name in ("run_mimii.py", "run_envmix.py"):
        source = (Path(__file__).resolve().parents[1] / "paper_benchmark" / name).read_text(encoding="utf-8")
        assert 'str(layer) for layer in config["layers"]' in source  # BEATs layers are ints: "|".join(ints) raises a TypeError


def test_the_common_subset_keeps_only_the_runs_every_beats_variant_completed():
    def rows(variant, method, keep):
        return [{"variant": variant, "method": method, "snr": s, "category": c, "machine_id": i, "img_roc_auc": 0.5 + 0.1 * (variant == "dinomaly_beats"), "f1_img": 0.6}
                for s, c, i in keep]

    all_runs = [(s, c, i) for s in ("0_db", "6_db", "m6_db") for c in tables.CATEGORIES for i in ("id_00", "id_02", "id_04", "id_06")]  # the 48 of the protocol
    partial = all_runs[:5]
    text = tables.common_subset_table(rows("dinomaly_beats", "dinomaly", all_runs) + rows("stfpm_beats", "stfpm", partial))
    assert text.count("| 5 |") == 2  # both variants restricted to the same 5 runs
    assert tables.common_subset_table(rows("dinomaly_beats", "dinomaly", all_runs)) == ""  # nothing incomplete: no extra table


def test_the_one_percent_variant_is_patchcore_with_a_memory_fraction_and_the_plan_can_pick_it():
    method, overrides = driver.VARIANTS["patchcore_beats_std_1pct"]
    assert method == "patchcore" and overrides["patchcore_memory_fraction"] == 0.01 and overrides["backbone"] == "BEATs"
    assert overrides["patchcore_band_banks"] and overrides["patchcore_score_std"] == "band"  # the same PatchCore as the 30000-patch one
    assert driver.plan(["mimii", "envmix"], ["patchcore_beats_std_1pct"]) == [("mimii", "patchcore_beats_std_1pct"), ("envmix", "patchcore_beats_std_1pct")]
    assert "patchcore_memory_fraction" not in driver.VARIANTS["patchcore_beats_std"][1]


def test_the_memory_is_a_share_of_the_patches_of_the_training_set():
    import torch
    from benchmark_common import patchcore_bank_size

    class Model:
        def batch_input(self, batch):
            return batch

        def embed(self, batch):  # [B, C, time, bands]
            return torch.zeros(batch.shape[0], 3, 62, 8)

    loader = torch.utils.data.DataLoader(torch.zeros(900, 10), batch_size=4)
    assert patchcore_bank_size(Model(), loader, 0.01) == int(900 * 62 * 8 * 0.01)  # 1% of the 446400 patches
    assert patchcore_bank_size(Model(), torch.utils.data.DataLoader(torch.zeros(2, 10), batch_size=2), 0.01) == 32  # never fewer than 4 per band
    import pytest

    with pytest.raises(ValueError):
        patchcore_bank_size(Model(), loader, 0.0)
