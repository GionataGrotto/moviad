"""Metrics of evaluation runs in the form of the DCASE results page: AUC (source), AUC (target), pAUC and the official score.

The AUC of a domain is the evaluator's one (the normal clips of the domain against ALL the anomalous clips), pAUC uses max_fpr 0.1 over all the clips.
Each column is the harmonic mean over the machines (the official score is the harmonic mean of all of them together).

    python paper_benchmark/eval_table.py tgt_pc_quota10:patchcore tgt_cmb_q10_mask:patchcore band_pc_banks_std_ls_train_all:patchcore
    python paper_benchmark/eval_table.py --machines tgt_pc_quota10:patchcore        # also one block per machine
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import hmean

import run_eval
from fuse_scores import read_machine_scores
from score_eval import evaluate


def harmonic(values) -> float:
    return float(hmean(np.maximum(np.array(values, dtype=float), 1e-12))) * 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", nargs="+", metavar="EXPERIMENT:METHOD")
    parser.add_argument("--machines", action="store_true", help="also print the three metrics of every machine")
    args = parser.parse_args()
    rows, detail = [], []
    for item in args.pairs:
        pair = tuple(item.split(":", 1))
        result = evaluate(read_machine_scores(run_eval.run_folder(pair)))
        source = [m["auc_source_official"] for m in result.values()]
        target = [m["auc_target_official"] for m in result.values()]
        pauc = [m["pauc_all"] for m in result.values()]
        rows.append(f"| {item} | {harmonic(source):.2f} | {harmonic(target):.2f} | {harmonic(pauc):.2f} | {run_eval.official_score(result):.2f} |")
        for machine, m in result.items():
            detail.append(f"| {item} | {machine} | {m['auc_source_official'] * 100:.2f} | {m['auc_target_official'] * 100:.2f} | {m['pauc_all'] * 100:.2f} |")
    print("| Run | AUC source (hmean) | AUC target (hmean) | pAUC (hmean) | Official score |")
    print("|---|---|---|---|---|")
    print("\n".join(rows))
    if args.machines:
        print("\n| Run | Machine | AUC source | AUC target | pAUC |")
        print("|---|---|---|---|---|")
        print("\n".join(detail))


if __name__ == "__main__":
    main()
