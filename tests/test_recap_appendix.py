import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_benchmark"))

import build_recap_appendix as appendix  # noqa: E402


def test_run_names_are_split_into_experiment_and_suffix():
    assert appendix.split_run_name("grid_beats_b12_ls_seed2") == ("beats_b12_ls", "_seed2")
    assert appendix.split_run_name("grid_beats_late_ls_frames") == ("beats_late_ls", "_frames")
    assert appendix.split_run_name("grid_baseline") == ("baseline", "")


def test_appendix_lists_results_pending_experiments_and_replaces_itself(tmp_path):
    dev = tmp_path / "dev" / "grid_baseline"
    dev.mkdir(parents=True)
    rows = [{"method": "knn", "machine": m, "auc_source": 0.6, "channel": 0, "backbone": "Cnn14", "layers": [], "seed": 1, "train_domain": "source"}
            for m in appendix.MACHINES]
    (dev / "dcase2026_task2_results.json").write_text(json.dumps(rows))
    lines = appendix.build(tmp_path / "dev", None)
    text = "\n".join(lines)
    assert "**0.600**" in text and "baseline" in text
    assert "`handcrafted`" in text  # an experiment without results is listed as pending
    recap = tmp_path / "recap.md"
    recap.write_text("# recap\n\nbody\n")
    appendix.write(recap, lines)
    appendix.write(recap, lines)
    result = recap.read_text()
    assert result.count(appendix.START) == 1 and result.startswith("# recap") and "body" in result
