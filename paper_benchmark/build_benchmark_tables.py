"""The MIMII and EnvMix tables of the paper (arXiv 2502.18328, Tables 1 and 3) with our Cnn14 reruns and our BEATs runs next to them.

    python paper_benchmark/build_benchmark_tables.py                       # prints both tables
    python paper_benchmark/build_benchmark_tables.py --write results/paper_benchmark/benchmark_tables.md

Rows: ``paper`` = the values printed in the paper (copied by hand, see PAPER_*); ``Cnn14`` = our runs with the paper's setting
(``results/paper_benchmark/{mimii,envmix}_results.csv``); ``BEATs`` = ``run_beats_benchmark.py`` (``results/paper_benchmark/beats``).
Every number of ours is the mean over all the runs of the CSV (SNR levels, machine ids / backgrounds, seeds), x 100, and the ``runs``
column says how many that is: the paper does not say how it aggregates, so the paper rows are not necessarily comparable run by run.
The best value of every column is in bold.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "paper_benchmark"

#: Table 1 of the paper (MIMII): method -> {category: (F1, ROC)}; the baseline has no F1
PAPER_MIMII = {
    "Baseline (DCASE 2020 AE)": {"fan": (None, 65.8), "pump": (None, 72.9), "slider": (None, 84.8), "valve": (None, 66.3)},
    "CFA": {"fan": (70.7, 65.7), "pump": (77.9, 76.0), "slider": (81.2, 79.9), "valve": (75.9, 73.2)},
    "PaDiM": {"fan": (69.2, 61.1), "pump": (72.3, 65.3), "slider": (79.0, 78.5), "valve": (76.4, 71.8)},
    "PatchCore": {"fan": (67.5, 59.6), "pump": (68.7, 59.3), "slider": (75.2, 76.2), "valve": (70.8, 67.1)},
    "STFPM": {"fan": (74.0, 72.5), "pump": (83.8, 85.9), "slider": (76.1, 74.4), "valve": (72.4, 66.3)},
}
CATEGORIES = ("fan", "pump", "slider", "valve")

#: Table 3 of the paper (EnvMix): method -> the seven numbers, in the order of ENVMIX_COLUMNS
PAPER_ENVMIX = {
    "CFA": (79.3, 76.4, 27.6, 56.0, 64.6, 46.5, 70.0),
    "PaDiM": (86.3, 81.8, 35.0, 56.9, 72.7, 55.0, 78.9),
    "PatchCore": (90.6, 85.8, 36.8, 58.3, 76.4, 59.9, 82.9),
    "STFPM": (53.2, 67.2, 19.4, 47.1, 47.0, 35.4, 47.9),
}
#: header, CSV column
ENVMIX_COLUMNS = (
    ("Sample ROC", "img_roc_auc"), ("Sample F1", "f1_img"), ("Spec. F1", "f1_pxl"), ("Spec. PRO", "au_pro_pxl"),
    ("Spec. ROC", "per_pixel_rocauc"), ("Temp. F1", "f1_tmp"), ("Temp. ROC", "tmp_auc_roc"),
)
METHOD_NAMES = {"cfa": "CFA", "padim": "PaDiM", "patchcore": "PatchCore", "stfpm": "STFPM", "dinomaly": "Dinomaly"}
VARIANT_LABELS = {
    "patchcore_beats_std": "PatchCore, BEATs + per-band banks + std", "padim_beats": "PaDiM, BEATs", "cfa_beats": "CFA, BEATs",
    "stfpm_beats": "STFPM, BEATs", "dinomaly_beats": "Dinomaly, BEATs",
}


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_of(rows: list[dict], column: str) -> float | None:
    values = []
    for row in rows:
        try:
            value = float(row.get(column, ""))
        except ValueError:
            continue
        if not math.isnan(value):
            values.append(value)
    return 100 * sum(values) / len(values) if values else None


def fmt(value: float | None, best: bool = False) -> str:
    if value is None:
        return "-"
    return f"**{value:.1f}**" if best else f"{value:.1f}"


def mark_best(table: list[list[float | None]]) -> list[list[bool]]:
    """For every column the rows that hold its best value."""
    flags = [[False] * len(row) for row in table]
    for column in range(len(table[0])):
        values = [row[column] for row in table if row[column] is not None]
        if values:
            top = max(values)
            for i, row in enumerate(table):
                flags[i][column] = row[column] is not None and abs(row[column] - top) < 1e-9
    return flags


def group_rows(rows: list[dict], method_key: str = "method") -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row.get("variant", ""), row[method_key]), []).append(row)
    return groups


def render(headers: list[str], labels: list[str], runs: list[str], table: list[list[float | None]]) -> str:
    flags = mark_best(table)
    lines = ["| Method | " + " | ".join(headers) + " | runs |", "|---|" + "---|" * (len(headers) + 1)]
    for label, count, values, row_flags in zip(labels, runs, table, flags):
        lines.append(f"| {label} | " + " | ".join(fmt(v, b) for v, b in zip(values, row_flags)) + f" | {count} |")
    return "\n".join(lines)


def mimii_table(cnn14_rows: list[dict], beats_rows: list[dict]) -> str:
    headers = [f"{category.capitalize()} {metric}" for category in CATEGORIES for metric in ("F1", "ROC")]
    labels, runs, table = [], [], []
    for name, values in PAPER_MIMII.items():
        labels.append(f"{name} [paper]")
        runs.append("-")
        table.append([values[c][i] for c in CATEGORIES for i in (0, 1)])
    for tag, rows in (("Cnn14, our rerun", cnn14_rows), ("", beats_rows)):
        for (variant, method), group in sorted(group_rows(rows).items(), key=lambda kv: list(METHOD_NAMES).index(kv[0][1]) if kv[0][1] in METHOD_NAMES else 99):
            line = []
            for category in CATEGORIES:
                selected = [r for r in group if r.get("category") == category]
                line += [mean_of(selected, "f1_img"), mean_of(selected, "img_roc_auc")]
            labels.append(f"{METHOD_NAMES.get(method, method)} [{tag}]" if tag else f"{VARIANT_LABELS.get(variant, variant)} [BEATs]")
            runs.append(str(len(group)))
            table.append(line)
    return render(headers, labels, runs, table)


TEMPORAL_COLUMNS = ("f1_tmp", "tmp_auc_roc", "pr_auc_tmp")


def valid_temporal(rows: list[dict]) -> list[dict]:
    """The rows whose temporal metrics are real. The old PatchCore rows of ``envmix_results.csv`` were written with the faithfulness
    values in the ``f1_tmp`` / ``tmp_auc_roc`` columns (an F1 cannot be negative): their temporal metrics were never stored."""
    good = []
    for row in rows:
        try:
            good.append(row) if float(row.get("f1_tmp", "")) >= 0 else None
        except ValueError:
            continue
    return good


def envmix_table(cnn14_rows: list[dict], beats_rows: list[dict]) -> str:
    headers = [name for name, _ in ENVMIX_COLUMNS]
    labels, runs, table = [], [], []
    for name, values in PAPER_ENVMIX.items():
        labels.append(f"{name} [paper]")
        runs.append("-")
        table.append(list(values))
    for tag, rows in (("Cnn14, our rerun", cnn14_rows), ("", beats_rows)):
        for (variant, method), group in sorted(group_rows(rows).items(), key=lambda kv: list(METHOD_NAMES).index(kv[0][1]) if kv[0][1] in METHOD_NAMES else 99):
            labels.append(f"{METHOD_NAMES.get(method, method)} [{tag}]" if tag else f"{VARIANT_LABELS.get(variant, variant)} [BEATs]")
            runs.append(str(len(group)))
            table.append([mean_of(valid_temporal(group) if column in TEMPORAL_COLUMNS else group, column) for _, column in ENVMIX_COLUMNS])
    return render(headers, labels, runs, table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cnn14-mimii", type=Path, default=RESULTS / "mimii_results.csv")
    parser.add_argument("--cnn14-envmix", type=Path, default=RESULTS / "envmix_results.csv")
    parser.add_argument("--beats-dir", type=Path, default=RESULTS / "beats")
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()
    text = "\n\n".join([
        "## MIMII (F1 and ROC AUC, x 100; mean over SNR levels, machine ids and seeds)",
        mimii_table(read_rows(args.cnn14_mimii), read_rows(args.beats_dir / "mimii_results.csv")),
        "## EnvMix (x 100; mean over backgrounds, SNR levels and seeds)",
        envmix_table(read_rows(args.cnn14_envmix), read_rows(args.beats_dir / "envmix_results.csv")),
    ])
    print(text)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten to {args.write}")


if __name__ == "__main__":
    main()
