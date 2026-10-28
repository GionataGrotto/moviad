"""Run every DCASE 2026 experiment in sequence, surviving failures, then fuse the results.

    uv run python paper_benchmark/run_all.py --base-config paper_benchmark/config.knn.json

Each (experiment, method) pair of ``PLAN`` is one separate run of ``run_dcase_grid.py``, so a crash
(out of memory, a missing checkpoint...) only loses that pair. A failed pair is retried
``--retries`` times, the others go on, and the failures are written to
``results/paper_benchmark/all_failures.json`` and printed at the end. Pairs already complete for
all 7 machines are skipped (``--skip-done``), so the same command resumes an interrupted job: run it
again, optionally with ``--only-failed`` to retry just what failed last time.

After the runs, the fusions of ``FUSIONS`` are computed from the saved scores of the runs that exist
and written to ``results/paper_benchmark/fusion_dcase2026.md``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from run_dcase_grid import write_summary  # noqa: E402
from fuse_scores import fuse, machine_auc, mean_auc, read_machine_scores  # noqa: E402

#: (experiment, method, run suffix, seed); ordered from the most useful to the most speculative
PLAN: list[tuple[str, str, str, int | None]] = [
    # references on BEATs
    ("baseline", "knn", "", None),
    ("beats_b12", "cfa", "", None),
    ("beats_b12", "patchcore", "", None),
    ("band_pc_banks_std", "patchcore", "", None),
    ("beats_late", "padim", "", None),
    ("beats_padim_full_cov", "padim", "", None),
    # CFA varies a lot between seeds: three more, to average them
    ("beats_b12", "cfa", "_seed1", 1),
    ("beats_b12", "cfa", "_seed2", 2),
    ("beats_b12", "cfa", "_seed3", 3),
    # handcrafted signal statistics (no pretrained model)
    ("handcrafted", "handcrafted", "", None),
    ("handcrafted_64", "handcrafted", "", None),
    ("knn_b12", "knn", "", None),
    # far-microphone least-squares denoising
    ("beats_late_ls", "patchcore", "", None),
    ("beats_late_ls_half", "patchcore", "", None),
    ("beats_late_ls", "cfa", "", None),
    ("knn_ls", "knn", "", None),
    # follow-ups of the LS result (round 2)
    ("handcrafted_128", "handcrafted", "", None),
    ("handcrafted_ls", "handcrafted", "", None),
    ("handcrafted_64_ls", "handcrafted", "", None),
    ("knn_b12_ls", "knn", "", None),
    ("band_pc_banks_std_ls", "patchcore", "", None),
    ("beats_b12_ls", "patchcore", "", None),
    ("beats_padim_full_cov_ls", "padim", "", None),
    # round 3: robustness of the LS winners (3 more seeds, which also carry the frames<N>pct aggregation),
    # fine-tuning on the LS signal, handcrafted settings
    ("beats_late_ls", "patchcore", "_frames", None),
    ("band_pc_banks_std_ls", "patchcore", "_seed1", 1),
    ("band_pc_banks_std_ls", "patchcore", "_seed2", 2),
    ("band_pc_banks_std_ls", "patchcore", "_seed3", 3),
    ("knn_ft_ls", "knn_ft", "", None),
    ("knn_ft", "knn_ft", "_seed1", 1),
    ("knn_ft", "knn_ft", "_seed2", 2),
    ("knn_ft_b12_ls", "knn_ft", "", None),
    ("handcrafted_96_ls", "handcrafted", "", None),
    ("handcrafted_64_ls_shrink0p3", "handcrafted", "", None),
    ("handcrafted_64_ls_shrink0p03", "handcrafted", "", None),
    # round 4: finer LS, LS on the other methods, richer handcrafted, stronger fine-tuning, several BEATs blocks
    ("band_pc_banks_std_ls_win", "patchcore", "", None),
    ("band_pc_banks_std_ls_global", "patchcore", "", None),
    ("handcrafted_64_ls_win", "handcrafted", "", None),
    ("handcrafted_64_ls_global", "handcrafted", "", None),
    ("knn_ls_win", "knn", "", None),
    ("knn_ls_global", "knn", "", None),
    ("beats_late_ls", "cfa", "_seed1", 1),
    ("beats_late_ls", "cfa", "_seed2", 2),
    ("beats_b12_ls", "cfa", "", None),
    ("stfpm_ls", "stfpm", "", None),
    ("dinomaly_ls", "dinomaly", "", None),
    ("handcrafted_64_ls_kurt", "handcrafted", "", None),
    ("handcrafted_64_ls_quant", "handcrafted", "", None),
    ("handcrafted_64_ls_corr", "handcrafted", "", None),
    ("handcrafted_64_ls_allstats", "handcrafted", "", None),
    ("handcrafted_64_ls_knn", "handcrafted", "", None),
    ("handcrafted_64_ls_iforest", "handcrafted", "", None),
    ("knn_ft_ls_strong", "knn_ft", "", None),
    ("knn_ft_ls_pseudo", "knn_ft", "", None),
    ("knn_b10_12_ls", "knn", "", None),
    ("knn_b8_10_12_ls", "knn", "", None),
    # round 5: PatchCore re-weighting of every patch
    ("beats_late_ls_rw", "patchcore", "", None),
    ("band_pc_banks_ls", "patchcore", "", None),
    ("band_pc_banks_ls_rw", "patchcore", "", None),
    ("band_pc_banks_std_ls_rw", "patchcore", "", None),
    # STFPM: optimiser and BEATs student
    ("baseline", "stfpm", "", None),
    ("stfpm_sgd_lr0p1", "stfpm", "", None),
    ("stfpm_adamw_1e4", "stfpm", "", None),
    ("stfpm_beats_late_adamw_1e5", "stfpm", "", None),
    ("stfpm_beats_b12_adamw_1e5", "stfpm", "", None),
    # LoRA + ArcFace fine-tuning of BEATs (the least tested part)
    ("knn_ft", "knn_ft", "", None),
    ("knn_ft_b12", "knn_ft", "", None),
]

#: the time-percentage pooling, on top of the --aggregation-sweep ones
EXTRA_AGGREGATIONS = ["frames10pct", "frames25pct", "frames50pct"]
Run = tuple[str, str, str]  # experiment, method, suffix
#: name -> runs whose scores are averaged (rank per machine, equal weights)
#: an entry may itself be a list of runs (e.g. seeds): those are averaged first and count as one component
FUSIONS: dict[str, list] = {
    "cfa_seeds": [("beats_b12", "cfa", s) for s in ("", "_seed1", "_seed2", "_seed3")],
    "cfa+patchcore": [("beats_b12", "cfa", ""), ("band_pc_banks_std", "patchcore", "")],
    "cfa+patchcore+knn": [("beats_b12", "cfa", ""), ("band_pc_banks_std", "patchcore", ""), ("baseline", "knn", "")],
    "patchcore+knn": [("band_pc_banks_std", "patchcore", ""), ("baseline", "knn", "")],
    "cfa_seeds+patchcore+knn": [("beats_b12", "cfa", s) for s in ("", "_seed1", "_seed2", "_seed3")]
    + [("band_pc_banks_std", "patchcore", ""), ("baseline", "knn", "")],
    "knn+handcrafted": [("baseline", "knn", ""), ("handcrafted", "handcrafted", "")],
    "cfa+patchcore+knn+handcrafted": [("beats_b12", "cfa", ""), ("band_pc_banks_std", "patchcore", ""),
                                      ("baseline", "knn", ""), ("handcrafted", "handcrafted", "")],
    "ls: patchcore+knn": [("beats_late_ls", "patchcore", ""), ("knn_ls", "knn", "")],
    "ls: patchcore+knn+handcrafted_64": [("beats_late_ls", "patchcore", ""), ("knn_ls", "knn", ""), ("handcrafted_64", "handcrafted", "")],
    "ls: patchcore+knn+handcrafted_64_ls": [("beats_late_ls", "patchcore", ""), ("knn_ls", "knn", ""), ("handcrafted_64_ls", "handcrafted", "")],
    "ls: patchcore+knn+cfa": [("beats_late_ls", "patchcore", ""), ("knn_ls", "knn", ""), ("beats_late_ls", "cfa", "")],
    # decided before seeing round 3: equal weights; seeds of the same method are averaged first by listing them
    "r3: pcb_ls + hc64_ls": [("band_pc_banks_std_ls", "patchcore", ""), ("handcrafted_64_ls", "handcrafted", "")],
    "r3: pcb_ls(4 seeds) + hc64_ls": [[("band_pc_banks_std_ls", "patchcore", s) for s in ("", "_seed1", "_seed2", "_seed3")],
                                      ("handcrafted_64_ls", "handcrafted", "")],
    "r3: pcb_ls + hc64_ls + knn_ft_ls": [("band_pc_banks_std_ls", "patchcore", ""), ("handcrafted_64_ls", "handcrafted", ""), ("knn_ft_ls", "knn_ft", "")],
    "r3: pcb_ls + hc64_ls + knn_ft_ls + knn_ls": [("band_pc_banks_std_ls", "patchcore", ""), ("handcrafted_64_ls", "handcrafted", ""),
                                                  ("knn_ft_ls", "knn_ft", ""), ("knn_ls", "knn", "")],
    # decided before seeing round 4
    "r4: pcb_ls + hc64_ls + knn_ft_ls_pseudo": [("band_pc_banks_std_ls", "patchcore", ""), ("handcrafted_64_ls", "handcrafted", ""),
                                                ("knn_ft_ls_pseudo", "knn_ft", "")],
    "r4: pcb_ls + hc64_ls_allstats + knn_ft_ls": [("band_pc_banks_std_ls", "patchcore", ""), ("handcrafted_64_ls_allstats", "handcrafted", ""),
                                                  ("knn_ft_ls", "knn_ft", "")],
    "r4: pcb_ls + hc64_ls + knn_b8_10_12_ls": [("band_pc_banks_std_ls", "patchcore", ""), ("handcrafted_64_ls", "handcrafted", ""),
                                               ("knn_b8_10_12_ls", "knn", "")],
    "r4: cfa_ls(3 seeds) + pcb_ls + hc64_ls": [[("beats_late_ls", "cfa", s) for s in ("", "_seed1", "_seed2")],
                                               ("band_pc_banks_std_ls", "patchcore", ""), ("handcrafted_64_ls", "handcrafted", "")],
    "knn+knn_ft": [("baseline", "knn", ""), ("knn_ft", "knn_ft", "")],
}

SUMMARY = ROOT / "results" / "paper_benchmark" / "grid_summary.md"
SCRATCH_SUMMARY = ROOT / "results" / "paper_benchmark" / "grid_summary_last_pair.md"
FAILURES = ROOT / "results" / "paper_benchmark" / "all_failures.json"
FUSION_REPORT = ROOT / "results" / "paper_benchmark" / "fusion_dcase2026.md"


def run_folder(run: Run, root: Path = ROOT) -> Path:
    experiment, method, suffix = run
    return root / "dcase2026_task2_evaluator" / "dev" / f"grid_{experiment}{suffix}" / "teams" / "moviad" / method


def pair_label(pair) -> str:
    experiment, method, suffix, _ = pair
    return f"{experiment}{suffix}:{method}"


def command_for(pair, args) -> list[str]:
    experiment, method, suffix, seed = pair
    command = [sys.executable, str(HERE / "run_dcase_grid.py"), "--base-config", str(args.base_config),
               "--experiments", experiment, "--methods", method, "--aggregation-sweep",
               "--extra-aggregations", *EXTRA_AGGREGATIONS, "--epochs", str(args.epochs), "--summary", str(SCRATCH_SUMMARY)]
    if not getattr(args, "rerun", False):
        command.append("--skip-done")
    if suffix:
        command += ["--run-suffix", suffix]
    if seed is not None:
        command += ["--seed", str(seed)]
    return command


def run_pair(pair, args, runner=subprocess.run) -> bool:
    """Run one pair, retrying on failure; True when a run exited cleanly."""
    for attempt in range(args.retries + 1):
        print(f"\n##### {pair_label(pair)} (attempt {attempt + 1}/{args.retries + 1})", flush=True)
        if runner(command_for(pair, args), cwd=ROOT).returncode == 0:
            return True
        print(f"!! {pair_label(pair)} failed", file=sys.stderr, flush=True)
        time.sleep(args.retry_wait)
    return False


def write_global_summary(path: Path = SUMMARY) -> None:
    """One table with every experiment and method of the plan, read from the saved scores."""
    experiments = list(dict.fromkeys(e for e, _, _, _ in PLAN))
    methods = list(dict.fromkeys(m for _, m, _, _ in PLAN))
    write_summary(path, experiments, methods)


def complete(run: Run, root: Path = ROOT) -> bool:
    folder = run_folder(run, root)
    return folder.exists() and len(read_machine_scores(folder)) == 7


def _runs(entry) -> list[Run]:
    return list(entry) if isinstance(entry, list) else [entry]


def _label(entry) -> str:
    return "+".join(f"{e}{s}:{m}" for e, m, s in _runs(entry))


def _component(entry, root: Path):
    scores = {_label(run_entry): read_machine_scores(run_folder(run_entry, root)) for run_entry in _runs(entry)}
    return scores if len(scores) == 1 else fuse(scores)[0]


def write_fusions(root: Path = ROOT, report: Path = FUSION_REPORT, plan_runs=None) -> list[str]:
    """Fuse every fusion whose runs are all complete; returns the report lines."""
    lines = ["# Fusioni DCASE 2026 (rank medio per macchina, pesi uguali, AUC source)", "",
             "| Fusione | Componenti | AUC fusa | AUC migliore componente |", "|---|---|---|---|"]
    detail = []
    for name, entries in FUSIONS.items():
        missing = [r for entry in entries for r in _runs(entry) if not complete(r, root)]
        if missing:
            lines.append(f"| {name} | {len(entries)} | n/d (mancano: {', '.join(f'{e}{s}:{m}' for e, m, s in missing)}) | |")
            continue
        inputs = {_label(entry): _component(entry, root) if len(_runs(entry)) > 1 else read_machine_scores(run_folder(_runs(entry)[0], root))
                  for entry in entries}
        fused, _ = fuse(inputs)
        best = max(mean_auc(scores)[0] for scores in inputs.values())
        fused_auc = mean_auc(fused)[0]
        lines.append(f"| {name} | {len(entries)} | {fused_auc:.3f} | {best:.3f} |")
        detail.append(f"- {name}: " + ", ".join(f"{m} {machine_auc(s)[0]:.3f}" for m, s in sorted(fused.items())))
    lines += ["", "Per macchina:", ""] + detail
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--retries", type=int, default=1, help="extra attempts of a failed pair")
    parser.add_argument("--retry-wait", type=float, default=30.0, help="seconds before a retry (lets the GPU memory free up)")
    parser.add_argument("--only-failed", action="store_true", help="run only the pairs listed in the last failure report")
    parser.add_argument("--only", nargs="+", default=None, help="run only pairs whose label contains one of these strings")
    parser.add_argument("--rerun", action="store_true", help="redo every pair, even those already complete")
    parser.add_argument("--summary-only", action="store_true", help="rebuild grid_summary.md and the fusions from the saved scores, run nothing")
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    args = parser.parse_args()

    if args.summary_only:
        write_global_summary()
        print(f"summary written: {SUMMARY}")
        if not args.skip_fusion:
            print("\n".join(write_fusions()))
        return
    plan = PLAN
    if args.only_failed:
        previous = set(json.loads(FAILURES.read_text(encoding="utf-8"))) if FAILURES.exists() else set()
        plan = [p for p in plan if pair_label(p) in previous]
    if args.only:
        plan = [p for p in plan if any(token in pair_label(p) for token in args.only)]
    if args.list:
        print("\n".join(pair_label(p) for p in plan))
        return

    failed = []
    for index, pair in enumerate(plan, 1):
        print(f"\n===== {index}/{len(plan)}: {pair_label(pair)}", flush=True)
        try:
            ok = run_pair(pair, args)
        except Exception as error:  # a launcher problem must not stop the remaining pairs
            print(f"!! {pair_label(pair)}: {error!r}", file=sys.stderr, flush=True)
            ok = False
        if not ok:
            failed.append(pair_label(pair))
        write_global_summary()
        FAILURES.parent.mkdir(parents=True, exist_ok=True)
        FAILURES.write_text(json.dumps(failed, indent=2), encoding="utf-8")
    if not args.skip_fusion:
        print("\n".join(write_fusions()))
    print(f"\nDone. Failed pairs ({len(failed)}): {failed if failed else 'none'}")
    if failed:
        print(f"Retry them with: --only-failed (list in {FAILURES})")


if __name__ == "__main__":
    main()
