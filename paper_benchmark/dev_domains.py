"""Source AUC, target AUC and the official-style score of dev runs that were tested on both domains.

    python paper_benchmark/dev_domains.py a=<run>/teams/moviad/patchcore b=<run2>/teams/moviad/patchcore

The dev score files carry the domain and the label in their names, so no ground truth is needed. The official-style score
is the harmonic mean over the machines of the source AUC, the target AUC and the pAUC (all clips), x 100, as the evaluator
computes it. Dev runs made with ``test_domain: source`` have no target clips and are reported with the source AUC only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import hmean

from score_dcase_csv import evaluate_folder


def dev_domain_metrics(folder: Path) -> dict:
    """``{"source": {machine: auc}, "target": {...}, "pauc": {...}, "official": float | None}`` of a score folder."""
    per = {name: {m: v[0] for m, v in evaluate_folder(folder, name).items() if v[0] is not None} for name in ("source", "target")}
    pauc = {m: v[1] for m, v in evaluate_folder(folder, "all").items() if v[1] is not None}
    machines = sorted(set(per["source"]) & set(per["target"]) & set(pauc))
    official = None
    if machines:
        values = [per["source"][m] for m in machines] + [per["target"][m] for m in machines] + [pauc[m] for m in machines]
        official = float(hmean(np.maximum(values, 1e-12))) * 100
    return {"source": per["source"], "target": per["target"], "pauc": pauc, "official": official}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="folder or name=folder")
    args = parser.parse_args()
    print(f"{'run':44s} source  target  pAUC   official  machines")
    for item in args.runs:
        name, sep, folder = item.partition("=")
        metrics = dev_domain_metrics(Path(folder if sep else item).expanduser())
        mean = lambda d: f"{np.mean(list(d.values())):.3f}" if d else "  -  "  # noqa: E731
        official = "   -  " if metrics["official"] is None else f"{metrics['official']:6.1f}"
        print(f"{(name if sep else Path(item).name):44s} {mean(metrics['source'])}   {mean(metrics['target'])}   {mean(metrics['pauc'])}  {official}    {len(metrics['source'])}")


if __name__ == "__main__":
    main()
