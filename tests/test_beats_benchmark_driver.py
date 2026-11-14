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


def test_the_tables_hold_the_paper_rows_ours_and_bold_the_best(tmp_path):
    def write(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    mimii_old = [{"method": "patchcore", "category": c, "img_roc_auc": 0.7, "f1_img": 0.7} for c in tables.CATEGORIES]
    mimii_new = [{"variant": "patchcore_beats_std", "method": "patchcore", "category": c, "img_roc_auc": 0.95, "f1_img": 0.9} for c in tables.CATEGORIES]
    text = tables.mimii_table(mimii_old, mimii_new)
    assert "PatchCore [paper]" in text and "PatchCore [Cnn14, our rerun]" in text and "[BEATs]" in text
    new_line = [line for line in text.splitlines() if "BEATs" in line][0]
    assert "**95.0**" in new_line and "**90.0**" in new_line  # best of its columns
    envmix_new = [{"variant": "padim_beats", "method": "padim", **{column: 0.5 for _, column in tables.ENVMIX_COLUMNS}}]
    assert "PaDiM, BEATs [BEATs]" in tables.envmix_table([], envmix_new)


def test_envmix_temporal_metrics_of_rows_with_shifted_columns_are_not_averaged():
    def row(method, f1_tmp, tmp_auc):
        base = {column: 0.5 for _, column in tables.ENVMIX_COLUMNS}
        return {**base, "method": method, "f1_tmp": f1_tmp, "tmp_auc_roc": tmp_auc}

    text = tables.envmix_table([row("patchcore", -0.1, 0.06), row("cfa", 0.6, 0.7)], [])
    patch = [line for line in text.splitlines() if "PatchCore [Cnn14" in line][0]
    cfa = [line for line in text.splitlines() if "CFA [Cnn14" in line][0]
    assert patch.split("|")[-4].strip() == "-" and patch.split("|")[-3].strip() == "-"  # temporal F1 / ROC missing, not -10 and 6
    assert "60.0" in cfa and "70.0" in cfa
