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

Run = tuple[str, str, str]  # experiment, method, suffix
#: name -> runs whose scores are averaged (rank per machine, equal weights)
FUSIONS: dict[str, list[Run]] = {
    "cfa_seeds": [("beats_b12", "cfa", s) for s in ("", "_seed1", "_seed2", "_seed3")],
    "cfa+patchcore": [("beats_b12", "cfa", ""), ("band_pc_banks_std", "patchcore", "")],
    "cfa+patchcore+knn": [("beats_b12", "cfa", ""), ("band_pc_banks_std", "patchcore", ""), ("baseline", "knn", "")],
    "patchcore+knn": [("band_pc_banks_std", "patchcore", ""), ("baseline", "knn", "")],
    "cfa_seeds+patchcore+knn": [("beats_b12", "cfa", s) for s in ("", "_seed1", "_seed2", "_seed3")]
    + [("band_pc_banks_std", "patchcore", ""), ("baseline", "knn", "")],
    "knn+handcrafted": [("baseline", "knn", ""), ("handcrafted", "handcrafted", "")],
    "cfa+patchcore+knn+handcrafted": [("beats_b12", "cfa", ""), ("band_pc_banks_std", "patchcore", ""),
                                      ("baseline", "knn", ""), ("handcrafted", "handcrafted", "")],
    "knn+knn_ft": [("baseline", "knn", ""), ("knn_ft", "knn_ft", "")],
}

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
               "--experiments", experiment, "--methods", method, "--skip-done", "--aggregation-sweep",
               "--epochs", str(args.epochs)]
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


def complete(run: Run, root: Path = ROOT) -> bool:
    folder = run_folder(run, root)
    return folder.exists() and len(read_machine_scores(folder)) == 7


def write_fusions(root: Path = ROOT, report: Path = FUSION_REPORT, plan_runs=None) -> list[str]:
    """Fuse every fusion whose runs are all complete; returns the report lines."""
    lines = ["# Fusioni DCASE 2026 (rank medio per macchina, pesi uguali, AUC source)", "",
             "| Fusione | Componenti | AUC fusa | AUC migliore componente |", "|---|---|---|---|"]
    detail = []
    for name, runs in FUSIONS.items():
        missing = [r for r in runs if not complete(r, root)]
        if missing:
            lines.append(f"| {name} | {len(runs)} | n/d (mancano: {', '.join(f'{e}{s}:{m}' for e, m, s in missing)}) | |")
            continue
        inputs = {f"{e}{s}:{m}": read_machine_scores(run_folder((e, m, s), root)) for e, m, s in runs}
        fused, _ = fuse(inputs)
        best = max(mean_auc(scores)[0] for scores in inputs.values())
        fused_auc = mean_auc(fused)[0]
        lines.append(f"| {name} | {len(runs)} | {fused_auc:.3f} | {best:.3f} |")
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
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    args = parser.parse_args()

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
        FAILURES.parent.mkdir(parents=True, exist_ok=True)
        FAILURES.write_text(json.dumps(failed, indent=2), encoding="utf-8")
    if not args.skip_fusion:
        print("\n".join(write_fusions()))
    print(f"\nDone. Failed pairs ({len(failed)}): {failed if failed else 'none'}")
    if failed:
        print(f"Retry them with: --only-failed (list in {FAILURES})")


if __name__ == "__main__":
    main()
