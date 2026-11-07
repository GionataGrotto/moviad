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
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from scipy.stats import hmean  # noqa: E402

from fuse_scores import fuse, read_machine_scores  # noqa: E402
from run_dcase_grid import EXPERIMENTS  # noqa: E402
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
    # with LS and the 10 target clips in the training set (the official protocol): see TARGET_PAIRS
    ("band_pc_banks_std_ls_train_all", "patchcore"): 0.706,
    ("beats_padim_full_cov_ls_train_all", "padim"): 0.655,
    ("beats_b12_ls_train_all", "cfa"): 0.649,
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
#: repeats of an extra with another seed (``--extra-seeds``): (name with the seed, method) -> (experiment, seed)
SEEDED: dict = {}
FUSION_FILE = ROOT / "results" / "paper_benchmark" / "eval_fusions.json"
#: (trained on source + target, trained on source only): what the target clips change, on the source and on the target AUC
TARGET_PAIRS = [
    (("band_pc_banks_std_ls_train_all", "patchcore"), ("band_pc_banks_std_ls", "patchcore")),
    (("beats_padim_full_cov_ls_train_all", "padim"), ("beats_padim_full_cov_ls", "padim")),
    (("beats_b12_ls_train_all", "cfa"), ("beats_b12_ls", "cfa")),
]
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
    seed = None
    if pair in SEEDED:
        experiment, seed = SEEDED[pair]
    command = [sys.executable, str(HERE / "run_dcase_grid.py"), "--base-config", str(args.base_config), "--experiments", experiment,
               "--methods", method, "--run-suffix", (f"_seed{seed}" if seed is not None else "") + SUFFIX, "--epochs", str(args.epochs),
               "--no-summary"]
    if seed is not None:
        command += ["--seed", str(seed)]
    for attempt in range(args.retries + 1):
        print(f"\n##### {experiment}:{method} (attempt {attempt + 1}/{args.retries + 1})", flush=True)
        if runner(command, cwd=ROOT).returncode == 0:
            return True
        time.sleep(args.retry_wait)
    return False


def discover_pairs(root: Path = ROOT) -> None:
    """Add to ``PAIRS`` every ``grid_<name>_eval`` run found on disk, so that the report lists them all (extras, repeats with other seeds)."""
    base = root / "dcase2026_task2_evaluator" / "dev"
    for folder in sorted(base.glob(f"grid_*{SUFFIX}")):
        name = folder.name[len("grid_") : -len(SUFFIX)]
        for method_dir in sorted((folder / "teams" / "moviad").glob("*")):
            pair = (name, method_dir.name)
            if "@" not in method_dir.name and pair not in DEV_AUC:
                DEV_AUC[pair] = None
                PAIRS.append(pair)


def load_fusions() -> dict[str, str]:
    return json.loads(FUSION_FILE.read_text(encoding="utf-8")) if FUSION_FILE.exists() else {}


def parse_fusion(spec: str) -> list[list[tuple[str, str]]]:
    """``"a:patchcore + b:knn_ft + c:cfa&c_seed1:cfa"`` -> components; ``&`` joins runs that are averaged first (seeds of one method)."""
    return [[tuple(pair.strip().split(":", 1)) for pair in component.split("&")] for component in spec.split("+")]


def fuse_spec(spec: str, scores: dict):
    """Rank-average fusion of a spec, or None if one of its runs has no complete scores."""
    components = parse_fusion(spec)
    if any(pair not in scores for component in components for pair in component):
        return None
    parts = {}
    for index, component in enumerate(components):
        label = "&".join(f"{e}:{m}" for e, m in component)
        parts[label] = scores[component[0]] if len(component) == 1 else fuse({f"{e}:{m}": scores[(e, m)] for e, m in component})[0]
    return fuse(parts)[0]


def seed_groups(results: dict) -> list[tuple[str, list]]:
    """Runs of the same experiment that differ only by ``_seedN``: ``[(base name:method, [pairs])]`` with at least two runs."""
    groups: dict = {}
    for pair in results:
        match = re.match(r"^(.*)_seed\d+$", pair[0])
        groups.setdefault((match.group(1) if match else pair[0], pair[1]), []).append(pair)
    return [(f"{name}:{method}", pairs) for (name, method), pairs in groups.items() if len(pairs) > 1]


def official_score(result: dict) -> float:
    """The evaluator's score: harmonic mean over the machines of the source AUC, the target AUC and the pAUC (x 100).

    The AUC of a domain is that of its normal clips against all the anomalous clips (source and target), as in the official script.
    """
    values = ([m["auc_source_official"] for m in result.values()] + [m["auc_target_official"] for m in result.values()]
              + [m["pauc_all"] for m in result.values()])
    return float(hmean(np.maximum(np.array(values, dtype=float), 1e-12))) * 100


def dev_text(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def dev_diff(with_ls, without) -> str:
    a, b = DEV_AUC[with_ls], DEV_AUC[without]
    return "-" if a is None or b is None else f"{a - b:+.3f}"


def build_report(root: Path = ROOT, gt_root: Path = GT_ROOT) -> list[str]:
    machines = eval_machines(gt_root)
    discover_pairs(root)
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
             "## Configurazioni singole", "", f"| Esperimento | Metodo | AUC source | AUC target | Punteggio ufficiale | AUC dev | {columns} |", "|---|---|---|---|---|---|" + "---|" * len(machines)]
    for pair in PAIRS:
        if pair in results:
            lines.append(f"| {pair[0]} | {pair[1]} | {mean_metric(results[pair]):.3f} | {mean_metric(results[pair], 'auc_target'):.3f} | "
                         f"{official_score(results[pair]):.1f} | {dev_text(DEV_AUC[pair])} | {per_machine(results[pair])} |")
        else:
            lines.append(f"| {pair[0]} | {pair[1]} | n/d | | | {dev_text(DEV_AUC[pair])} | |")
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
    seeds = seed_groups(results)
    if seeds:
        lines += ["", "## Rumore tra seed (stessa configurazione, seed diversi)", "",
                  "| Configurazione | Run | AUC source (media) | AUC target (media) | Punteggio ufficiale: media | minimo - massimo | scarto tipo |", "|---|---|---|---|---|---|---|"]
        for label, pairs in seeds:
            official = [official_score(results[p]) for p in pairs]
            lines.append(f"| {label} | {len(pairs)} | {np.mean([mean_metric(results[p]) for p in pairs]):.3f} | "
                         f"{np.mean([mean_metric(results[p], 'auc_target') for p in pairs]):.3f} | {np.mean(official):.1f} | "
                         f"{min(official):.1f} - {max(official):.1f} | {np.std(official):.2f} |")
    requested = load_fusions()
    if requested:
        lines += ["", "## Fusioni richieste (rank medio per macchina, pesi uguali tra i componenti; `&` = media dei seed dello stesso metodo)", "",
                  f"| Fusione | AUC source | AUC target | Punteggio ufficiale | {columns} |", "|---|---|---|---|" + "---|" * len(machines)]
        for label, spec in requested.items():
            combined = fuse_spec(spec, scores)
            if combined is None:
                lines.append(f"| {label} | n/d | | | |")
                continue
            result = evaluate(combined, gt_root)
            lines.append(f"| {label} | {mean_metric(result):.3f} | {mean_metric(result, 'auc_target'):.3f} | {official_score(result):.1f} | {per_machine(result)} |")
    lines += ["", "## Con i 10 clip target nel training (protocollo ufficiale) e senza", "",
              "| Metodo | Con target: source | Con target: target | Con target: ufficiale | Senza: source | Senza: target | Senza: ufficiale | Differenza ufficiale |",
              "|---|---|---|---|---|---|---|---|"]
    for with_target, without in TARGET_PAIRS:
        if with_target in results and without in results:
            a_, b_ = results[with_target], results[without]
            lines.append(f"| {with_target[1]} ({with_target[0]}) | {mean_metric(a_):.3f} | {mean_metric(a_, 'auc_target'):.3f} | {official_score(a_):.1f} | "
                         f"{mean_metric(b_):.3f} | {mean_metric(b_, 'auc_target'):.3f} | {official_score(b_):.1f} | {official_score(a_) - official_score(b_):+.1f} |")
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
    parser.add_argument("--extra", nargs="+", default=[], metavar="EXPERIMENT:METHOD",
                        help="more grid experiments to run on the evaluation set and add to the report (e.g. tgt_pc_quota25:patchcore)")
    parser.add_argument("--extra-seeds", nargs="+", type=int, default=[], metavar="SEED",
                        help="also run every --extra with these seeds (suffix _seedN); the base run is the one with the default seed")
    parser.add_argument("--fusion", nargs="+", default=[], metavar="LABEL=SPEC",
                        help="a fusion to report, e.g. 'pc+knn=tgt_pc_quota10:patchcore + knn_ft_ls_both:knn_ft'; runs joined by & are averaged "
                             "first (seeds); it is saved in results/paper_benchmark/eval_fusions.json and stays in the report")
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from the saved scores, run nothing")
    args = parser.parse_args()

    for item in args.extra:  # chosen on the dev machines beforehand, run here once
        experiment, _, method = item.partition(":")
        if experiment not in EXPERIMENTS or not method:
            parser.error(f"--extra expects EXPERIMENT:METHOD with an experiment of run_dcase_grid.py, got {item!r}")
        if (experiment, method) not in DEV_AUC:
            DEV_AUC[(experiment, method)] = None
            PAIRS.append((experiment, method))
        for seed in args.extra_seeds:
            pair = (f"{experiment}_seed{seed}", method)
            SEEDED[pair] = (experiment, seed)
            if pair not in DEV_AUC:
                DEV_AUC[pair] = None
                PAIRS.append(pair)
    if args.fusion:
        saved = load_fusions()
        for item in args.fusion:
            label, sep, spec = item.partition("=")
            if not sep:
                parser.error(f"--fusion expects LABEL=SPEC, got {item!r}")
            saved[label.strip()] = spec.strip()
        FUSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        FUSION_FILE.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
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
