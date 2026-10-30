"""Write the appendix of the DCASE recap: every experiment that has results, and the ones that have none yet.

    python paper_benchmark/build_recap_appendix.py --dev-dir ~/Downloads/dev --eval-scores ~/Downloads/eval_scores

``--dev-dir`` holds ``grid_<name>/dcase2026_task2_results.json`` (one per dev run, as copied from the machine that ran
the grid); ``--eval-scores`` holds ``dcase2026_task2_evaluator/dev/grid_<name>_eval/teams/moviad/<method>`` (the
evaluation-set scores). The appendix replaces what is between its two markers in
``results/paper_benchmark/recap_dcase2026.md`` (it is added at the end the first time), so it can be regenerated whenever
new results arrive.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import hmean

from fuse_scores import read_machine_scores
from run_dcase_grid import EXPERIMENTS, MACHINES
from score_eval import evaluate, eval_machines

ROOT = Path(__file__).resolve().parent.parent
RECAP = ROOT / "results" / "paper_benchmark" / "recap_dcase2026.md"
START = "<!-- ALL-EXPERIMENTS-START -->"
END = "<!-- ALL-EXPERIMENTS-END -->"
_SUFFIX = re.compile(r"^(?P<name>.+?)(?P<suffix>(_seed\d+|_frames)*)$")


def split_run_name(run: str) -> tuple[str, str]:
    """``grid_beats_b12_ls_seed2`` -> (``beats_b12_ls``, ``_seed2``): the experiment and the suffix of the run."""
    name = run[5:] if run.startswith("grid_") else run
    match = _SUFFIX.match(name)
    return (match["name"], match["suffix"]) if match and match["name"] in EXPERIMENTS else (name, "")


def describe(name: str) -> str:
    return EXPERIMENTS[name][0] if name in EXPERIMENTS else "(experiment no longer defined)"


def dev_rows(dev_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(Path(dev_dir).glob("grid_*/dcase2026_task2_results.json")):
        run = path.parent.name
        if run.endswith("_eval"):
            continue
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        by_method: dict[str, dict] = {}
        for record in records:
            if record.get("auc_source") is not None:
                by_method.setdefault(record["method"], {})[record["machine"]] = record
        for method, machines in by_method.items():
            aucs = [machines[m]["auc_source"] for m in MACHINES if m in machines]
            name, suffix = split_run_name(run)
            first = next(iter(machines.values()))
            extras = {}
            for agg in ("mean", "top10pct", "max"):
                values = [machines[m]["extra_aggregations"][agg]["auc_source"] for m in machines if agg in machines[m].get("extra_aggregations", {})]
                if len(values) == len(machines):
                    extras[agg] = float(np.mean(values))
            rows.append({"experiment": name, "suffix": suffix, "method": method, "n": len(aucs), "auc": float(np.mean(aucs)),
                         "per_machine": [machines[m]["auc_source"] if m in machines else None for m in MACHINES],
                         "channel": first.get("channel"), "backbone": first.get("backbone"), "layers": first.get("layers"),
                         "seed": first.get("seed"), "train": first.get("train_domain"), "extras": extras})
    return rows


def eval_rows(eval_root: Path) -> list[dict]:
    machines = eval_machines()
    rows = []
    dev = Path(eval_root) / "dcase2026_task2_evaluator" / "dev"
    for run in sorted(dev.glob("grid_*_eval")):
        for method_dir in sorted((run / "teams" / "moviad").glob("*")):
            if "@" in method_dir.name:
                continue
            scores = read_machine_scores(method_dir)
            if not set(machines) <= set(scores):
                continue
            result = evaluate(scores)
            mean = lambda key: float(np.mean([result[m][key] for m in machines]))
            values = [result[m]["auc_source"] for m in machines] + [result[m]["auc_target"] for m in machines] + [result[m]["pauc_all"] for m in machines]
            rows.append({"experiment": split_run_name(run.name[: -len("_eval")])[0], "method": method_dir.name,
                         "source": mean("auc_source"), "target": mean("auc_target"), "all": mean("auc_all"), "pauc": mean("pauc_all"),
                         "official": float(hmean(np.maximum(np.array(values), 1e-12))) * 100,
                         "per_machine": [result[m]["auc_source"] for m in machines]})
    return rows


def fmt(value) -> str:
    return "-" if value is None else f"{value:.2f}"


def build(dev_dir: Path, eval_root: Path | None) -> list[str]:
    rows = dev_rows(dev_dir)
    lines = [START, "", "## Appendice: tutti gli esperimenti", "",
             "Generata da `paper_benchmark/build_recap_appendix.py` a partire dai risultati salvati.", "",
             "### Dev (AUC source, 7 macchine, training sul solo source salvo dove indicato)", "",
             "| Esperimento | Metodo | Seed | AUC | media mappa | top10% | max | " + " | ".join(MACHINES) + " | n | canale | backbone/blocchi | training |",
             "|---|---|---|---|---|---|---|" + "---|" * len(MACHINES) + "---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["method"], -r["auc"])):
        label = f"{r['experiment']}{r['suffix']}: {describe(r['experiment'])}"
        extra = " | ".join(f"{r['extras'][a]:.3f}" if a in r["extras"] else "-" for a in ("mean", "top10pct", "max"))
        layers = f"{r['backbone']} {r['layers']}" if r["backbone"] else "-"
        lines.append(f"| {label} | {r['method']} | {r['seed']} | **{r['auc']:.3f}** | {extra} | " + " | ".join(fmt(v) for v in r["per_machine"])
                     + f" | {r['n']} | {r['channel']} | {layers} | {r['train']} |")
    ran = {r["experiment"] for r in rows}
    pending_names = sorted(name for name in EXPERIMENTS if name not in ran)
    lines += ["", "### Esperimenti definiti senza risultati nella cartella dev fornita", "",
             "Non e' detto che non siano stati eseguiti: quelli dei primi giri (livelli di Cnn14, normalizzazione dello spettrogramma, "
             "near - far, train_all, STFPM a pesi casuali...) hanno i risultati nelle sezioni precedenti di questo documento, e la cartella "
             "dev usata per generare questa appendice non li contiene piu'. Gli altri sono da eseguire.", ""]
    lines += [f"- `{name}`: {describe(name)}" for name in pending_names] or ["- nessuno"]
    if eval_root is not None:
        ev = eval_rows(eval_root)
        machines = eval_machines()
        lines += ["", "### Evaluation set (training sul solo source; AUC source, target, tutte, pAUC e punteggio ufficiale ricostruito)", "",
                  "| Esperimento | Metodo | AUC source | AUC target | AUC tutte | pAUC | Punteggio ufficiale | " + " | ".join(machines) + " |",
                  "|---|---|---|---|---|---|---|" + "---|" * len(machines)]
        for r in sorted(ev, key=lambda r: -r["source"]):
            lines.append(f"| {r['experiment']}: {describe(r['experiment'])} | {r['method']} | **{r['source']:.3f}** | {r['target']:.3f} | {r['all']:.3f} | "
                         f"{r['pauc']:.3f} | {r['official']:.1f} | " + " | ".join(fmt(v) for v in r["per_machine"]) + " |")
    lines += ["", END]
    return lines


def write(recap: Path, lines: list[str]) -> None:
    text = recap.read_text(encoding="utf-8")
    block = "\n".join(lines) + "\n"
    if START in text and END in text:
        before, rest = text.split(START, 1)
        after = rest.split(END, 1)[1].lstrip("\n")
        text = before + block + ("\n" + after if after else "")
    else:
        text = text.rstrip("\n") + "\n\n" + block
    recap.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev-dir", type=Path, required=True)
    parser.add_argument("--eval-scores", type=Path, default=None)
    parser.add_argument("--recap", type=Path, default=RECAP)
    args = parser.parse_args()
    lines = build(args.dev_dir.expanduser(), args.eval_scores.expanduser() if args.eval_scores else None)
    write(args.recap, lines)
    print(f"appendix written to {args.recap} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
