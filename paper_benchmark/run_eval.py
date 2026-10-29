"""Run the chosen configurations on the DCASE 2026 Task 2 evaluation set and report the source-domain AUC.

    uv run python paper_benchmark/run_eval.py --base-config paper_benchmark/config.eval.json

``config.eval.json`` is the usual config with ``dcase2026.dataset_path`` pointing at the folder that holds
``<machine>/train`` (additional training dataset) and ``<machine>/test`` (evaluation clips). Training is on the
source domain only and the AUC is on the source clips of the test set (the domain of each clip comes from the
ground truth shipped with the evaluator), as in the dev experiments. Nothing is tuned here: the report shows

  1. the configuration fixed in advance (FIXED_FUSION),
  2. the configuration that was best on the dev machines, applied here (dev-selected),
  3. the best of everything tried, chosen by its evaluation AUC (oracle: optimistic by construction).

Every pair runs in its own process, is retried once and skipped if its scores already exist, as in run_all.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from fuse_scores import fuse, read_machine_scores  # noqa: E402
from score_eval import GT_ROOT, eval_machines, evaluate, mean_metric  # noqa: E402

SUFFIX = "_eval"
REPORT = ROOT / "results" / "paper_benchmark" / "eval_dcase2026.md"
SCRATCH = ROOT / "results" / "paper_benchmark" / "eval_last_pair.md"

#: (experiment, method) -> source AUC on the 7 dev machines (2026-10-28) or None if it was never run on dev.
#: Chosen: the best variant of every model family with dev AUC above 0.65 (Dinomaly and CFA as the best of their family),
#: plus the components of the best fusions, plus the same configurations without LS to check the LS.
DEV_AUC = {
    # with LS
    ("band_pc_banks_std_ls", "patchcore"): 0.705,
    ("handcrafted_96_ls", "handcrafted"): 0.724,
    ("handcrafted_64_ls", "handcrafted"): 0.708,
    ("knn_ft_ls", "knn_ft"): 0.722,
    ("knn_ft_ls_pseudo", "knn_ft"): 0.688,
    ("knn_b8_10_12_ls", "knn"): 0.701,
    ("beats_padim_full_cov_ls", "padim"): 0.656,
    ("beats_b12_ls", "cfa"): 0.620,
    ("dinomaly_ls", "dinomaly"): 0.546,
    # without LS: the check of the LS
    ("band_pc_banks_std", "patchcore"): 0.638,
    ("handcrafted_96", "handcrafted"): None,
    ("knn_ft", "knn_ft"): 0.687,
    ("knn_b8_10_12", "knn"): None,
    ("beats_padim_full_cov", "padim"): 0.600,
    ("beats_b12", "cfa"): 0.657,
    ("baseline", "dinomaly"): 0.535,
}
PAIRS = list(DEV_AUC)
DEV_FIXED_NO_LS: list = []  # pairs without LS that are needed even when the LS check is skipped (none)
FIXED_FUSION = "PatchCore LS + handcrafted_64 LS + knn_ft LS"
#: name -> (components, dev AUC of the fusion or None); all of them were fixed before any evaluation result was seen
FUSIONS = {
    FIXED_FUSION: ([("band_pc_banks_std_ls", "patchcore"), ("handcrafted_64_ls", "handcrafted"), ("knn_ft_ls", "knn_ft")], 0.747),
    "PatchCore LS + handcrafted_64 LS + knn_ft LS (pseudo-classes)": (
        [("band_pc_banks_std_ls", "patchcore"), ("handcrafted_64_ls", "handcrafted"), ("knn_ft_ls_pseudo", "knn_ft")], 0.751),
    "PatchCore LS + handcrafted_64 LS": ([("band_pc_banks_std_ls", "patchcore"), ("handcrafted_64_ls", "handcrafted")], 0.725),
    "PatchCore LS + handcrafted_96 LS + knn_ft LS (the best of each family)": (
        [("band_pc_banks_std_ls", "patchcore"), ("handcrafted_96_ls", "handcrafted"), ("knn_ft_ls", "knn_ft")], None),
    "one per family, all with LS (7 components)": (
        [("band_pc_banks_std_ls", "patchcore"), ("handcrafted_96_ls", "handcrafted"), ("knn_ft_ls", "knn_ft"), ("knn_b8_10_12_ls", "knn"),
         ("beats_padim_full_cov_ls", "padim"), ("beats_b12_ls", "cfa"), ("dinomaly_ls", "dinomaly")], None),
    "stessa del migliore per famiglia, senza LS": (
        [("band_pc_banks_std", "patchcore"), ("handcrafted_96", "handcrafted"), ("knn_ft", "knn_ft")], None),
}
#: (with LS, without LS): the LS check
LS_PAIRS = [
    (("band_pc_banks_std_ls", "patchcore"), ("band_pc_banks_std", "patchcore")),
    (("handcrafted_96_ls", "handcrafted"), ("handcrafted_96", "handcrafted")),
    (("knn_ft_ls", "knn_ft"), ("knn_ft", "knn_ft")),
    (("knn_b8_10_12_ls", "knn"), ("knn_b8_10_12", "knn")),
    (("beats_padim_full_cov_ls", "padim"), ("beats_padim_full_cov", "padim")),
    (("beats_b12_ls", "cfa"), ("beats_b12", "cfa")),
    (("dinomaly_ls", "dinomaly"), ("baseline", "dinomaly")),
]


def run_folder(pair, root: Path = ROOT) -> Path:
    experiment, method = pair
    return root / "dcase2026_task2_evaluator" / "dev" / f"grid_{experiment}{SUFFIX}" / "teams" / "moviad" / method


def is_done(pair, machines: list[str], root: Path = ROOT) -> bool:
    folder = run_folder(pair, root)
    return folder.exists() and set(machines) <= set(read_machine_scores(folder))


def run_pair(pair, args, runner=subprocess.run) -> bool:
    experiment, method = pair
    command = [sys.executable, str(HERE / "run_dcase_grid.py"), "--base-config", str(args.base_config), "--experiments", experiment,
               "--methods", method, "--run-suffix", SUFFIX, "--epochs", str(args.epochs), "--summary", str(SCRATCH)]
    for attempt in range(args.retries + 1):
        print(f"\n##### {experiment}:{method} (attempt {attempt + 1}/{args.retries + 1})", flush=True)
        if runner(command, cwd=ROOT).returncode == 0:
            return True
        time.sleep(args.retry_wait)
    return False


def dev_text(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def dev_diff(with_ls, without) -> str:
    a, b = DEV_AUC[with_ls], DEV_AUC[without]
    return "-" if a is None or b is None else f"{a - b:+.3f}"


def build_report(root: Path = ROOT, gt_root: Path = GT_ROOT) -> list[str]:
    machines = eval_machines(gt_root)
    scores, results = {}, {}
    for pair in PAIRS:
        if is_done(pair, machines, root):
            scores[pair] = read_machine_scores(run_folder(pair, root))
            results[pair] = evaluate(scores[pair], gt_root)
    fused = {}
    for name, (parts, _) in FUSIONS.items():
        if all(p in scores for p in parts):
            combined = fuse({f"{e}:{m}": scores[(e, m)] for e, m in parts})[0]
            fused[name] = evaluate(combined, gt_root)

    def per_machine(result):
        return " | ".join(f"{result[m]['auc_source']:.2f}" if m in result else "-" for m in machines)

    columns = " | ".join(machines)
    lines = ["# Evaluation set DCASE 2026 Task 2: AUC source (training sul solo source)", "",
             f"Macchine: {', '.join(machines)}. AUC calcolata sui clip source del test con le etichette del valutatore.", "",
             "## Configurazioni singole", "", f"| Esperimento | Metodo | AUC source | AUC dev | {columns} |", "|---|---|---|---|" + "---|" * len(machines)]
    for pair in PAIRS:
        if pair in results:
            lines.append(f"| {pair[0]} | {pair[1]} | {mean_metric(results[pair]):.3f} | {dev_text(DEV_AUC[pair])} | {per_machine(results[pair])} |")
        else:
            lines.append(f"| {pair[0]} | {pair[1]} | n/d | {dev_text(DEV_AUC[pair])} | |")
    lines += ["", "## Fusioni (rank medio per macchina, pesi uguali)", "", f"| Fusione | AUC source | AUC dev | {columns} |", "|---|---|---|" + "---|" * len(machines)]
    for name, (_, dev) in FUSIONS.items():
        label = name + (" (fissata in anticipo)" if name == FIXED_FUSION else "")
        dev_cell = dev_text(dev)
        lines.append(f"| {label} | " + (f"{mean_metric(fused[name]):.3f} | {dev_cell} | {per_machine(fused[name])} |" if name in fused else f"n/d | {dev_cell} | |"))
    lines += ["", "## Verifica del LS (stessa configurazione con e senza)", "",
              "| Con LS | Senza LS | AUC con | AUC senza | Differenza | Differenza sul dev |", "|---|---|---|---|---|---|"]
    for with_ls, without in LS_PAIRS:
        if with_ls in results and without in results:
            a, b = mean_metric(results[with_ls]), mean_metric(results[without])
            lines.append(f"| {with_ls[0]}:{with_ls[1]} | {without[0]}:{without[1]} | {a:.3f} | {b:.3f} | {a - b:+.3f} | {dev_diff(with_ls, without)} |")
    singles = {f"{p[0]}:{p[1]}": mean_metric(r) for p, r in results.items()}
    fusion_aucs = {n: mean_metric(r) for n, r in fused.items()}
    lines += ["", "## Tarata e non tarata", ""]
    if FIXED_FUSION in fusion_aucs:
        lines.append(f"- **Fissata in anticipo** ({FIXED_FUSION}): {fusion_aucs[FIXED_FUSION]:.3f}")
    if results:
        known = [p for p in results if DEV_AUC[p] is not None]
        best = max(known, key=lambda p: DEV_AUC[p])
        lines.append(f"- **Migliore sul dev tra le singole** ({best[0]}:{best[1]}, dev {DEV_AUC[best]:.3f}): {mean_metric(results[best]):.3f}")
    with_dev = [n for n in fusion_aucs if FUSIONS[n][1] is not None]
    if with_dev:
        best_fusion = max(with_dev, key=lambda n: FUSIONS[n][1])
        lines.append(f"- **Migliore sul dev tra le fusioni** ({best_fusion}, dev {FUSIONS[best_fusion][1]:.3f}): {fusion_aucs[best_fusion]:.3f}")
    everything = {**singles, **fusion_aucs}
    if everything:
        top = max(everything, key=everything.get)
        lines.append(f"- **Migliore guardando l'evaluation (oracolo, ottimistico)**: {top}: {everything[top]:.3f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-wait", type=float, default=30.0)
    parser.add_argument("--skip-ls-check", action="store_true", help="do not run the configurations without LS")
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from the saved scores, run nothing")
    args = parser.parse_args()

    machines = eval_machines()
    failed = []
    if not args.report_only:
        plan = [p for p in PAIRS if not (args.skip_ls_check and "_ls" not in p[0] and p not in DEV_FIXED_NO_LS)]
        for index, pair in enumerate(plan, 1):
            print(f"\n===== {index}/{len(plan)}: {pair[0]}:{pair[1]}", flush=True)
            if is_done(pair, machines):
                print("already done, skipped")
                continue
            try:
                ok = run_pair(pair, args)
            except Exception as error:  # a launcher problem must not stop the other pairs
                print(f"!! {pair}: {error!r}", file=sys.stderr, flush=True)
                ok = False
            if not ok:
                failed.append(f"{pair[0]}:{pair[1]}")
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            REPORT.write_text("\n".join(build_report()) + "\n", encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = build_report()
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nFailed pairs ({len(failed)}): {failed if failed else 'none'}\nReport: {REPORT}")


if __name__ == "__main__":
    main()
