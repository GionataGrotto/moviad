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
    for runs in run_all.FUSIONS.values():
        assert set(runs) <= planned


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
