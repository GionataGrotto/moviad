import types

from paper_benchmark import run_all


def args(**kw):
    base = dict(base_config="c.json", epochs=30, retries=1, retry_wait=0)
    return types.SimpleNamespace(**{**base, **kw})


def test_plan_and_fusions_reference_known_experiments_and_methods():
    from paper_benchmark.run_dcase_grid import EXPERIMENTS

    for experiment, method, suffix, seed in run_all.PLAN:
        assert experiment in EXPERIMENTS
        only = EXPERIMENTS[experiment][2]
        assert only is None or method in only, (experiment, method)
    planned = {(e, m, s) for e, m, s, _ in run_all.PLAN}
    for entries in run_all.FUSIONS.values():
        flat = {r for entry in entries for r in (entry if isinstance(entry, list) else [entry])}
        assert flat <= planned


def test_command_has_suffix_and_seed():
    command = run_all.command_for(("beats_b12", "cfa", "_seed2", 2), args())
    assert command[command.index("--run-suffix") + 1] == "_seed2" and command[command.index("--seed") + 1] == "2"
    assert "--skip-done" in command


def test_failed_pair_is_retried_then_reported():
    calls = []

    def runner(command, cwd=None):
        calls.append(command)
        return types.SimpleNamespace(returncode=1 if len(calls) < 2 else 0)

    pair = ("baseline", "knn", "", None)
    assert run_all.run_pair(pair, args(), runner) is True and len(calls) == 2
    calls.clear()
    assert run_all.run_pair(pair, args(retries=2), lambda c, cwd=None: calls.append(c) or types.SimpleNamespace(returncode=1)) is False
    assert len(calls) == 3


def test_rerun_drops_skip_done():
    pair = ("baseline", "knn", "", None)
    assert "--skip-done" in run_all.command_for(pair, args())
    assert "--skip-done" not in run_all.command_for(pair, args(rerun=True))


def test_pairs_write_their_summary_to_a_scratch_file():
    command = run_all.command_for(("baseline", "knn", "", None), args())
    assert command[command.index("--summary") + 1] == str(run_all.SCRATCH_SUMMARY)
    assert run_all.SCRATCH_SUMMARY != run_all.SUMMARY


def test_write_fusions_averages_a_group_of_seeds_as_one_component(tmp_path, monkeypatch):
    machines = ["ToyCar", "ToyCarEmu", "bearingEmu", "fan", "gearboxEmu", "sliderEmu", "valveEmu"]
    files = [f"section_00_source_test_{'anomaly' if i % 2 else 'normal'}_{i:04d}.wav" for i in range(10)]

    def write(run, offset):
        folder = run_all.run_folder(run, tmp_path)
        folder.mkdir(parents=True)
        for machine in machines:
            rows = "".join(f"{f},{i % 2 + 0.1 * ((i * 7 + offset) % 5)}\n" for i, f in enumerate(files))
            (folder / f"anomaly_score_{machine}_section_00_test.csv").write_text(rows)

    seeds = [("e", "m", ""), ("e", "m", "_s1")]
    for i, run in enumerate(seeds + [("h", "x", "")]):
        write(run, i)
    monkeypatch.setattr(run_all, "FUSIONS", {"grouped": [seeds, ("h", "x", "")], "missing": [("nope", "m", "")]})
    lines = run_all.write_fusions(tmp_path, tmp_path / "report.md")
    grouped = next(line for line in lines if line.startswith("| grouped"))
    assert "| 2 |" in grouped and "n/d" not in grouped
    assert "n/d (mancano: nope:m)" in next(line for line in lines if line.startswith("| missing"))


def test_shared_adapters_are_made_once_and_only_when_an_experiment_needs_them(tmp_path):
    calls = []

    def runner(command, cwd=None):
        calls.append(command)
        return types.SimpleNamespace(returncode=0)

    shared = ("ssl_pc_reg_ls_shared", "patchcore", "", None)
    assert run_all.ensure_adapter(shared, args(), runner, root=tmp_path) is True
    assert len(calls) == 1 and "train_beats_ssl.py" in calls[0][1] and calls[0][calls[0].index("--objective") + 1] == "reg"
    (tmp_path / "adapters").mkdir()
    (tmp_path / "adapters" / "ssl_reg_ls.pt").write_bytes(b"")
    assert run_all.ensure_adapter(shared, args(), runner, root=tmp_path) is True and len(calls) == 1  # the file exists: nothing to make
    assert run_all.ensure_adapter(("tgt_pc_quota25", "patchcore", "", None), args(), runner, root=tmp_path) is True and len(calls) == 1


def test_round_seven_is_in_the_plan():
    labels = {run_all.pair_label(p) for p in run_all.PLAN}
    assert "tgt_pc_quota25:patchcore" in labels and "ssl_pc_mask_ls:patchcore" in labels and "dino_beats_ls_tw:dinomaly" in labels


def test_the_plan_runs_the_riskiest_pairs_last():
    labels = [run_all.pair_label(p) for p in run_all.PLAN]
    position = {label: i for i, label in enumerate(labels)}
    assert position["tgt_pc_plain:patchcore"] < position["tgt_pc_quota25:patchcore"] < position["tgt_pc_pseudo_coral:patchcore"]
    assert position["tgt_pc_pseudo_coral:patchcore"] < position["ssl_pc_reg_ls:patchcore"] < position["dino_beats_ls:dinomaly"]
    assert position["dino_beats_ls:dinomaly"] < position["ssl_pc_reg_ls_shared:patchcore"] < position["dino_beats_ls_ssl:dinomaly"]
    assert position["dino_beats_ls_ssl_tw:dinomaly"] < position["stfpm_beats_b12_adamw_1e5:stfpm"] == len(labels) - 1
    assert position["baseline:knn"] < position["tgt_pc_source_only:patchcore"]  # what already ran stays in front
