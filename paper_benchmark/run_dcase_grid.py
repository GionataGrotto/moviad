"""One-factor-at-a-time experiments around the standard DCASE 2026 baseline.

Every experiment changes one thing with respect to the baseline (Cnn14 blocks 2-4, near
microphone, training on source only) and keeps the test protocol fixed (source-domain test
clips), so the mean source AUC of each row is directly comparable:

    baseline   conv_block2-4, channel 0, train on source
    layers_fine   conv_block1-3        finer time-frequency grid
    layers_deep   conv_block3-5        coarser grid, more abstract features
    train_all     + the 10 target training clips (the official 990+10 protocol)
    near_minus_far   waveform = channel 0 - channel 1 (cancels the shared noise)

    python paper_benchmark/run_dcase_grid.py --base-config paper_benchmark/config.knn.json
    python paper_benchmark/run_dcase_grid.py --base-config ... --experiments baseline layers_deep --dry-run

Each run is isolated under ``dcase2026_task2_evaluator/dev/<run_name>``. A summary table is
written to ``results/paper_benchmark/grid_summary.md`` after every run, so partial results
are kept if the job is interrupted.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from score_dcase_csv import evaluate_folder  # noqa: E402

LAYERS_BLOCKS_2_4 = ["conv_block2", "conv_block3", "conv_block4"]

#: name -> (description, config overrides, methods that must be skipped)
EXPERIMENTS = {
    "baseline": ("conv_block2-4, near mic, train source", {}, set()),
    "layers_fine": ("conv_block1-3", {"layers": ["conv_block1", "conv_block2", "conv_block3"]}, {"dinomaly", "draem"}),
    "layers_deep": ("conv_block3-5", {"layers": ["conv_block3", "conv_block4", "conv_block5"]}, {"dinomaly", "draem"}),
    "train_all": ("train source + 10 target clips", {"dcase2026": {"train_domains": "all"}}, set()),
    "near_minus_far": ("input = near - far", {"dcase2026": {"channel": "diff"}}, set()),
}
DEFAULT_METHODS = ["patchcore", "padim", "cfa", "stfpm", "dinomaly"]


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config(base: dict, name: str, epochs: int | None, aggregation: str = "temporal_topk_mean") -> dict:
    description, overrides, _ = EXPERIMENTS[name]
    config = deep_merge(base, {"backbone": "Cnn14", "layers": LAYERS_BLOCKS_2_4})
    config = deep_merge(config, overrides)
    # the protocol is fixed: always evaluate on the source-domain test clips
    config["dcase2026"] = deep_merge(config.get("dcase2026", {}), {"test_domain": "source", "run_name": f"grid_{name}", "score_aggregation": aggregation})
    if name != "train_all":
        config["dcase2026"].setdefault("train_domains", "source")
    if epochs is not None:
        config["epochs"] = epochs
    return config


def methods_for(name: str, methods: list[str]) -> list[str]:
    return [m for m in methods if m not in EXPERIMENTS[name][2]]


def collect(experiments: list[str], methods: list[str], domain: str = "source") -> dict:
    table = {}
    for name in experiments:
        run = ROOT / "dcase2026_task2_evaluator" / "dev" / f"grid_{name}" / "teams" / "moviad"
        for method in methods:
            folder = run / method
            results = evaluate_folder(folder, domain) if folder.exists() else {}
            aucs = [auc for auc, _, _ in results.values() if auc is not None]
            table[(name, method)] = (sum(aucs) / len(aucs), len(aucs)) if aucs else None
    return table


def write_summary(path: Path, experiments: list[str], methods: list[str]) -> None:
    table = collect(experiments, methods)
    lines = ["# Griglia DCASE 2026: AUC source media (7 macchine)", "",
             "| Esperimento | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for name in experiments:
        cells = []
        for method in methods:
            value = table[(name, method)]
            cells.append("n/d" if value is None else f"{value[0]:.3f}" + ("" if value[1] == 7 else f" ({value[1]} macchine)"))
        lines.append(f"| {name}: {EXPERIMENTS[name][0]} | " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", type=Path, required=True, help="config with dcase2026.dataset_path and the device")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--epochs", type=int, default=30, help="epochs of cfa/stfpm/dinomaly (patchcore and padim ignore it)")
    parser.add_argument("--aggregation", choices=("max", "mean", "temporal_topk_mean"), default="temporal_topk_mean",
                        help="file-level score; the same for every experiment")
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "paper_benchmark" / "grid_summary.md")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    base = json.loads(args.base_config.expanduser().read_text(encoding="utf-8"))
    config_dir = ROOT / "paper_benchmark" / "grid_configs"
    config_dir.mkdir(exist_ok=True)

    for name in args.experiments:
        methods = methods_for(name, args.methods)
        if not methods:
            continue
        config_path = config_dir / f"config.grid_{name}.json"
        config_path.write_text(json.dumps(build_config(base, name, args.epochs, args.aggregation), indent=2), encoding="utf-8")
        command = [sys.executable, str(HERE / "run_dcase2026_task2.py"), "--config", str(config_path), "--methods", *methods]
        print(f"\n=== {name}: {EXPERIMENTS[name][0]}\n$ {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            print(f"!! experiment {name} failed (exit code {result.returncode}); continuing", file=sys.stderr)
        write_summary(args.summary, args.experiments, args.methods)
        print(f"summary updated: {args.summary}", flush=True)


if __name__ == "__main__":
    main()
