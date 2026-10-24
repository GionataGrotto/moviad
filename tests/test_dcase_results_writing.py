import json

from paper_benchmark import run_dcase2026_task2 as runner
from paper_benchmark import run_dcase_grid as grid


def row(method, machine, auc, debug=False):
    return {"method": method, "machine": machine, "auc_source": auc, "debug": debug}


def test_a_new_invocation_keeps_the_rows_of_the_previous_ones():
    rows = [row("patchcore", "fan", 0.6), row("padim", "fan", 0.5)]

    rows = runner.upsert_result(rows, row("cfa", "fan", 0.55))
    assert [r["method"] for r in rows] == ["patchcore", "padim", "cfa"]

    rows = runner.upsert_result(rows, row("padim", "ToyCar", 0.58))  # another machine of the same method
    assert len(rows) == 4


def test_repeating_a_run_replaces_its_row_instead_of_duplicating_it():
    rows = [row("padim", "fan", 0.5), row("cfa", "fan", 0.55)]

    rows = runner.upsert_result(rows, row("padim", "fan", 0.62))

    assert sorted((r["method"], r["auc_source"]) for r in rows) == [("cfa", 0.55), ("padim", 0.62)]


def test_debug_rows_never_overwrite_real_ones():
    rows = runner.upsert_result([row("padim", "fan", 0.5)], row("padim", "fan", 0.9, debug=True))
    assert len(rows) == 2


def test_load_results_handles_missing_and_broken_files(tmp_path):
    path = tmp_path / "dcase2026_task2_results.json"
    assert runner.load_results(path) == []

    path.write_text(json.dumps([row("padim", "fan", 0.5)]))
    assert runner.load_results(path) == [row("padim", "fan", 0.5)]

    path.write_text("{ this is not json")
    assert runner.load_results(path) == []
    assert (tmp_path / "dcase2026_task2_results.corrupt.json").exists()  # the broken file is kept, not deleted


def write_scores(folder, machine, scores):
    folder.mkdir(parents=True, exist_ok=True)
    lines = [f"section_00_source_test_{label}_{i:04d}.wav,{score}" for i, (label, score) in enumerate(scores)]
    (folder / f"anomaly_score_{machine}_section_00_test.csv").write_text("\n".join(lines))


def test_summary_detail_lists_every_machine_of_every_run(tmp_path):
    run = tmp_path / "dcase2026_task2_evaluator" / "dev" / "grid_baseline" / "teams" / "moviad" / "padim"
    write_scores(run, "fan", [("normal", 0.1), ("normal", 0.2), ("anomaly", 0.8), ("anomaly", 0.9)])  # AUC 1.0
    write_scores(run, "ToyCar", [("normal", 0.8), ("normal", 0.9), ("anomaly", 0.1), ("anomaly", 0.2)])  # AUC 0.0

    lines = grid.machine_detail(["baseline", "layers_deep"], ["padim", "cfa"], root=tmp_path)

    text = "\n".join(lines)
    assert "### baseline" in text and "### layers_deep" not in text  # no files, no table
    padim_row = [line for line in lines if line.startswith("| padim")][0].split("|")
    cells = [c.strip() for c in padim_row[2:-1]]
    assert cells[grid.MACHINES.index("fan")] == "1.000" and cells[grid.MACHINES.index("ToyCar")] == "0.000"
    assert cells[grid.MACHINES.index("valveEmu")] == "-"  # a machine that was not run
    assert cells[-2] == "0.500"  # mean AUC of the two machines run


def test_skip_done_only_keeps_the_methods_without_all_seven_machines(tmp_path):
    base = tmp_path / "dcase2026_task2_evaluator" / "dev" / "grid_baseline" / "teams" / "moviad"
    for machine in grid.MACHINES:  # padim: complete
        write_scores(base / "padim", machine, [("normal", 0.1), ("normal", 0.2), ("anomaly", 0.8), ("anomaly", 0.9)])
    for machine in grid.MACHINES[:3]:  # cfa: interrupted after 3 machines
        write_scores(base / "cfa", machine, [("normal", 0.1), ("normal", 0.2), ("anomaly", 0.8), ("anomaly", 0.9)])

    pending = grid.pending_methods("baseline", ["padim", "cfa", "patchcore"], root=tmp_path)

    assert pending == ["cfa", "patchcore"]  # complete padim is skipped, partial cfa and missing patchcore run


def test_summary_file_contains_the_best_per_method_and_the_machine_detail(tmp_path):
    base = tmp_path / "dcase2026_task2_evaluator" / "dev"
    good = [("normal", 0.1), ("normal", 0.2), ("anomaly", 0.8), ("anomaly", 0.9)]  # AUC 1.0
    bad = [("normal", 0.8), ("normal", 0.9), ("anomaly", 0.1), ("anomaly", 0.2)]  # AUC 0.0
    for experiment, scores in (("baseline", bad), ("layers_deep", good)):
        for machine in grid.MACHINES:
            write_scores(base / f"grid_{experiment}" / "teams" / "moviad" / "padim", machine, scores)
    summary = tmp_path / "summary.md"

    grid.write_summary(summary, ["baseline", "layers_deep"], ["padim"], root=tmp_path)

    text = summary.read_text()
    assert "Migliore per metodo" in text and "- padim: layers_deep (1.000)" in text
    assert "## Dettaglio per macchina" in text and "### layers_deep:" in text and "### baseline:" in text
    assert "| padim | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |" in text
