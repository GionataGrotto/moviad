"""The final DCASE 2026 Task 2 runs on the evaluation set: the best configuration of every model, and the ablation of PatchCore.

    uv run python paper_benchmark/run_dcase_final.py --base-config paper_benchmark/config.eval.json --seeds 0 1 2

Everything is trained with the 10 target clips (the official protocol) unless the row says "solo source", evaluated with the
official definition of the score (the AUC of a domain = its normal clips against all the anomalous ones; harmonic mean over the
machines of source AUC, target AUC and pAUC) and reported as mean +- standard deviation over the seeds.

Parts (``--parts``):
  models    the best configuration of PatchCore, PaDiM, CFA, Dinomaly and STFPM (the choices were made on the dev machines, except
            STFPM, whose two candidates are both reported: nothing was selected on the evaluation set)
  ladder    PatchCore from the Cnn14 baseline, adding one piece at a time: BEATs, banks per band, score standardisation, LS, the 10 target clips, the quota
  leave     the best PatchCore with one piece taken away (quota, target clips, LS, standardisation, banks)

Every run is a process of ``run_dcase_grid.py`` (via ``run_eval.run_pair``); one that already has its scores is skipped, so the script can be
stopped and started again. Seed 0 is the default seed and reuses the runs already made by ``run_eval.py`` (same names). The report is
``results/paper_benchmark/final_dcase2026.md`` and is rewritten after every run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from scipy.stats import hmean  # noqa: E402

import run_eval  # noqa: E402
from fuse_scores import read_machine_scores  # noqa: E402
from run_dcase_grid import EXPERIMENTS  # noqa: E402
from score_eval import GT_ROOT, eval_machines, evaluate  # noqa: E402

REPORT = ROOT / "results" / "paper_benchmark" / "final_dcase2026.md"
PARTS = ("models", "ladder", "leave")


def plan(dinomaly: str = "mask_tw") -> dict[str, list[tuple[str, str, str]]]:
    """``{part: [(label, experiment, method)]}``."""
    dino = {"mask_tw": "dino_beats_ls_mask_tw", "tw": "dino_beats_ls_tw"}[dinomaly]
    return {
        "models": [
            ("PatchCore (LS, banche + std, quota target 10%)", "tgt_pc_quota10", "patchcore"),
            ("PaDiM (BEATs 8,10,12, covarianza completa, LS)", "beats_padim_full_cov_ls_train_all", "padim"),
            ("CFA (BEATs blocco 12, LS)", "cfa_b12_ls_both", "cfa"),
            (f"Dinomaly (BEATs, LS, target pesati{', adattatore mask' if dinomaly == 'mask_tw' else ''})", dino, "dinomaly"),
            ("STFPM (BEATs 8,10,12, LS)", "stfpm_beats_ls_both", "stfpm"),
            ("STFPM (BEATs blocco 12, LS)", "stfpm_beats_b12_ls_both", "stfpm"),
        ],
        "ladder": [
            ("0. Cnn14, canale vicino, solo source", "baseline", "patchcore"),
            ("1. + BEATs 8,10,12", "beats_late", "patchcore"),
            ("2. + una banca per banda", "band_pc_banks", "patchcore"),
            ("3. + standardizzazione dei punteggi", "band_pc_banks_std", "patchcore"),
            ("4. + LS (near - LS(far))", "band_pc_banks_std_ls", "patchcore"),
            ("5. + 10 clip target", "band_pc_banks_std_ls_train_all", "patchcore"),
            ("6. + quota target 10%", "tgt_pc_quota10", "patchcore"),
        ],
        "leave": [
            ("migliore (LS, banche, std, 10 target, quota 10%)", "tgt_pc_quota10", "patchcore"),
            ("senza quota", "band_pc_banks_std_ls_train_all", "patchcore"),
            ("senza clip target (solo source)", "band_pc_banks_std_ls", "patchcore"),
            ("senza LS", "abl_q10_no_ls", "patchcore"),
            ("senza standardizzazione", "abl_q10_no_std", "patchcore"),
            ("senza banche per banda e senza standardizzazione", "abl_q10_no_banks_std", "patchcore"),
        ],
    }


def run_pairs(parts, seeds, dinomaly="mask_tw") -> list[tuple[str, str, int]]:
    """Every distinct ``(experiment, method, seed)`` of the chosen parts, in the order of the plan."""
    wanted = []
    for part in parts:
        for _, experiment, method in plan(dinomaly)[part]:
            for seed in seeds:
                if (experiment, method, seed) not in wanted:
                    wanted.append((experiment, method, seed))
    return wanted


def pair_of(experiment: str, method: str, seed: int) -> tuple[str, str]:
    """The ``run_eval`` pair of a run: seed 0 is the default run (the name of ``run_eval.py``), the others carry ``_seedN``."""
    if seed == 0:
        return experiment, method
    pair = (f"{experiment}_seed{seed}", method)
    run_eval.SEEDED[pair] = (experiment, seed)
    return pair


def run_metrics(pair: tuple[str, str], gt_root: Path = GT_ROOT) -> dict[str, float] | None:
    """Source AUC, target AUC, pAUC (harmonic means over the machines) and the official score, x 100; ``None`` if the run is not complete."""
    machines = eval_machines(gt_root)
    if not run_eval.is_done(pair, machines):
        return None
    result = evaluate(read_machine_scores(run_eval.run_folder(pair)), gt_root)
    harmonic = lambda key: float(hmean(np.maximum([m[key] for m in result.values()], 1e-12))) * 100  # noqa: E731
    return {"source": harmonic("auc_source_official"), "target": harmonic("auc_target_official"), "pauc": harmonic("pauc_all"),
            "official": run_eval.official_score(result)}


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"))


def cell(runs: list[dict], key: str) -> str:
    if not runs:
        return "n/d"
    mean, std = mean_std([r[key] for r in runs])
    return f"{mean:.1f}" if np.isnan(std) else f"{mean:.1f} ± {std:.1f}"


def table(rows: list[tuple[str, list[dict]]], deltas: bool = False, reference: int | None = None) -> list[str]:
    """Markdown rows: label, seeds, source, target, pAUC, official (and the change of the official score).

    ``deltas`` compares every row with the previous one (the ladder); ``reference`` with the row of that index (leave-one-out).
    """
    compare = deltas or reference is not None
    header = ["Configurazione", "seed", "AUC source", "AUC target", "pAUC", "Punteggio ufficiale"] + (["Differenza"] if compare else [])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    means = [mean_std([r["official"] for r in runs])[0] if runs else None for _, runs in rows]
    for index, (label, runs) in enumerate(rows):
        cells = [label, str(len(runs)), cell(runs, "source"), cell(runs, "target"), cell(runs, "pauc"), cell(runs, "official")]
        if compare:
            base_index = index - 1 if deltas else reference
            valid = base_index is not None and base_index >= 0 and base_index != index and means[index] is not None and means[base_index] is not None
            cells.append(f"{means[index] - means[base_index]:+.1f}" if valid else "")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def build_report(parts, seeds, dinomaly="mask_tw", metrics=None) -> list[str]:
    """The markdown report; ``metrics`` maps ``(experiment, method, seed)`` to the metrics of a run (default: read from disk)."""
    def runs_of(experiment, method):
        found = []
        for seed in seeds:
            value = metrics.get((experiment, method, seed)) if metrics is not None else run_metrics(pair_of(experiment, method, seed))
            if value:
                found.append(value)
        return found

    lines = ["# DCASE 2026 Task 2, evaluation set: configurazioni finali e ablazione di PatchCore", "",
             f"Punteggio ufficiale (script del valutatore), 5 macchine, media ± deviazione standard su {len(seeds)} seed ({', '.join(map(str, seeds))}). "
             "Tutti i modelli sono addestrati con i 10 clip target (protocollo ufficiale) salvo dove scritto \"solo source\". "
             "AUC source, AUC target e pAUC sono medie armoniche sulle macchine, come nella tabella della pagina dei risultati.", ""]
    plans = plan(dinomaly)
    if "models" in parts:
        lines += ["## Migliore configurazione di ogni modello", ""]
        lines += table([(label, runs_of(e, m)) for label, e, m in plans["models"]])
        lines += ["", "Le configurazioni sono state scelte sul dev, tranne STFPM, di cui sono riportati entrambi i candidati: sull'evaluation non e' stato selezionato nulla.", ""]
    if "ladder" in parts:
        lines += ["## PatchCore: cosa aggiunge ogni pezzo (scala cumulativa)", ""]
        lines += table([(label, runs_of(e, m)) for label, e, m in plans["ladder"]], deltas=True)
        lines += ["", "La differenza e' rispetto alla riga precedente. I passi 0-4 sono addestrati sul solo source.", ""]
    if "leave" in parts:
        lines += ["## PatchCore: la migliore configurazione senza un pezzo", ""]
        lines += table([(label, runs_of(e, m)) for label, e, m in plans["leave"]], reference=0)
        lines += ["", "La differenza e' rispetto alla configurazione migliore. Gli effetti non si sommano (quota e augmentation insieme peggiorano), quindi questa tabella e la scala cumulativa vanno lette insieme.", ""]
    lines += ["Differenze sotto circa 1 punto non sono distinguibili dal rumore tra seed (evaluation: 0,1-0,2 per PatchCore; molto di piu' per CFA).", ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", type=Path, required=True, help="config with dcase2026.dataset_path pointing at the evaluation set")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument("--dinomaly", choices=("mask_tw", "tw"), default="mask_tw",
                        help="mask_tw: with the shared masked-patch adapter (best on dev); tw: without the adapter")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-wait", type=int, default=30)
    parser.add_argument("--report-only", action="store_true", help="do not run anything, only rewrite the report")
    parser.add_argument("--dry-run", action="store_true", help="list the runs still to do")
    args = parser.parse_args()
    runs = run_pairs(args.parts, args.seeds, args.dinomaly)
    for experiment, _, _ in runs:
        if experiment not in EXPERIMENTS:
            parser.error(f"unknown experiment {experiment!r} in run_dcase_grid.py")

    machines = eval_machines()
    todo = [(e, m, s) for e, m, s in runs if not run_eval.is_done(pair_of(e, m, s), machines)]
    print(f"{len(todo)} runs to do (of {len(runs)})", flush=True)
    if args.dry_run:
        for e, m, s in todo:
            print(f"  {e}:{m} seed {s}")
        return
    failed = []
    if not args.report_only:
        for index, (experiment, method, seed) in enumerate(todo, 1):
            print(f"\n===== {index}/{len(todo)}: {experiment}:{method} seed {seed}", flush=True)
            try:
                ok = run_eval.run_pair(pair_of(experiment, method, seed), args)
            except Exception as error:  # a launcher problem must not stop the other runs
                print(f"!! {experiment}:{method} seed {seed}: {error!r}", file=sys.stderr, flush=True)
                ok = False
            if not ok:
                failed.append(f"{experiment}:{method}:seed{seed}")
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            REPORT.write_text("\n".join(build_report(args.parts, args.seeds, args.dinomaly)) + "\n", encoding="utf-8")
    lines = build_report(args.parts, args.seeds, args.dinomaly)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nFailed runs ({len(failed)}): {failed if failed else 'none'}\nReport: {REPORT}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
