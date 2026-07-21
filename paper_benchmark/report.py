from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from benchmark_common import run_cli


METRIC_COLUMNS = [
    "img_roc_auc",
    "f1_img",
    "pr_auc_img",
    "per_pixel_rocauc",
    "f1_pxl",
    "pr_auc_pxl",
    "au_pro_pxl",
    "mse_pxl",
    "f1_tmp",
    "tmp_auc_roc",
    "pr_auc_tmp",
    "ff_v1",
    "ff_v2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate paper benchmark CSV files into a Markdown report.")
    parser.add_argument("--results", default="results/paper_benchmark")
    return parser.parse_args()


def read_rows(results_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(results_dir.glob("*_results.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                row["_source"] = csv_path.name
                rows.append(row)
    return rows


def as_float(value: str):
    if value in ("", "None", None):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return None if math.isnan(number) else number


def group_key(row: dict[str, str]) -> tuple[str, ...]:
    dataset = row.get("dataset", "")
    if dataset == "mimii":
        return (
            dataset,
            row.get("method", ""),
            row.get("snr", ""),
            row.get("category", ""),
        )
    if dataset == "envmix":
        return (
            dataset,
            row.get("method", ""),
            row.get("snr_db", ""),
            row.get("background_category", ""),
        )
    return (dataset, row.get("method", ""))


def summarize(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    summary = {}
    for key, group in groups.items():
        entry = {"runs": str(len(group))}
        for metric in METRIC_COLUMNS:
            values = [as_float(row.get(metric, "")) for row in group]
            values = [value for value in values if value is not None]
            if not values:
                continue
            avg = mean(values)
            spread = stdev(values) if len(values) > 1 else 0.0
            entry[metric] = f"{avg:.4f} +/- {spread:.4f}"
        summary[key] = entry
    return summary


def render_table(title: str, keys: list[tuple[str, ...]], summary: dict[tuple[str, ...], dict[str, str]]) -> list[str]:
    if not keys:
        return []
    available_metrics = [
        metric
        for metric in METRIC_COLUMNS
        if any(metric in summary[key] for key in keys)
    ]
    lines = [f"## {title}", ""]
    lines.append("| Group | Runs | " + " | ".join(available_metrics) + " |")
    lines.append("|---|---:|" + "|".join("---:" for _ in available_metrics) + "|")
    for key in keys:
        label = " / ".join(part for part in key if part)
        row = summary[key]
        values = [row.get(metric, "") for metric in available_metrics]
        lines.append(f"| {label} | {row['runs']} | " + " | ".join(values) + " |")
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results).resolve()
    rows = read_rows(results_dir)
    report_path = results_dir / "report.md"

    if not rows:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            "# Paper Benchmark Report\n\nNo result CSV files were found yet.\n",
            encoding="utf-8",
        )
        print(f"Wrote empty report to {report_path}")
        return

    summary = summarize(rows)
    mimii_keys = sorted(key for key in summary if key[0] == "mimii")
    envmix_keys = sorted(key for key in summary if key[0] == "envmix")

    lines = [
        "# Paper Benchmark Report",
        "",
        f"Result rows: {len(rows)}",
        "",
        "Values are reported as mean +/- sample standard deviation across matching runs.",
        "",
    ]
    lines.extend(render_table("MIMII", mimii_keys, summary))
    lines.extend(render_table("EnvMix", envmix_keys, summary))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    run_cli(main)
