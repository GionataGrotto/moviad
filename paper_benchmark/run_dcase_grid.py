"""One-factor-at-a-time experiments around the standard DCASE 2026 baseline.

Every experiment changes one thing with respect to the baseline (Cnn14 blocks 2-4, near
microphone, training on source only) and keeps the test protocol fixed (source-domain test
clips), so the mean source AUC of each row is directly comparable:

    baseline   conv_block2-4, channel 0, train on source
    layers_fine / layers_deep   conv_block1-3 / conv_block3-5
    train_all     + the 10 target training clips (the official 990+10 protocol)
    near_minus_far   waveform = channel 0 - channel 1 (cancels the shared noise)
    stfpm_random_student   student initialised at random, as in the STFPM paper
    spec_band_norm / spec_fill_dead_bands / spec_both   log-mel z-scored per band / bands above 8 kHz
                 filled with the last live band / both
    agg_mean     file score = mean of the anomaly map instead of the temporal top-k
    patchcore_standardize / patchcore_big_bank   z-scored channels / 60000-patch memory bank
    padim_full_cov   full covariance instead of the diagonal one
    cfa_fine_memory   gamma_c = 1 (no k-means reduction of the memory bank)
    dinomaly_small / dinomaly_base   larger ViT encoders
    epochs_long   100 epochs for cfa, stfpm and dinomaly

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

LAYER_METHODS = {"patchcore", "padim", "cfa", "stfpm"}
SPECTROGRAM_METHODS = LAYER_METHODS | {"dinomaly"}
#: name -> (description, config overrides, methods the experiment applies to; None = all)
EXPERIMENTS = {
    "baseline": ("conv_block2-4, near mic, train source", {}, None),
    "layers_fine": ("conv_block1-3", {"layers": ["conv_block1", "conv_block2", "conv_block3"]}, LAYER_METHODS),
    "layers_deep": ("conv_block3-5", {"layers": ["conv_block3", "conv_block4", "conv_block5"]}, LAYER_METHODS),
    "train_all": ("train source + 10 target clips", {"dcase2026": {"train_domains": "all"}}, None),
    "near_minus_far": ("input = near - far", {"dcase2026": {"channel": "diff"}}, None),
    "stfpm_random_student": ("STFPM student at random init", {"stfpm_student_pretrained": False}, {"stfpm"}),
    "patchcore_standardize": ("PatchCore z-scored channels", {"patchcore_standardize": True}, {"patchcore"}),
    "patchcore_big_bank": ("PatchCore bank 60000", {"memory_bank_size": 60000}, {"patchcore"}),
    "padim_full_cov": ("PaDiM full covariance", {"padim_diag_cov": False}, {"padim"}),
    "cfa_fine_memory": ("CFA gamma_c=1", {"cfa_gamma_c": 1}, {"cfa"}),
    "dinomaly_small": ("Dinomaly deit_small", {"dinomaly_encoder": "deit_small_patch16_224.fb_in1k"}, {"dinomaly"}),
    "dinomaly_base": ("Dinomaly deit_base", {"dinomaly_encoder": "deit_base_patch16_224.fb_in1k"}, {"dinomaly"}),
    "epochs_long": ("100 epochs", {"epochs": 100}, {"cfa", "stfpm", "dinomaly"}),
    "spec_band_norm": ("log-mel z-scored per band", {"spec_band_norm": "zscore"}, SPECTROGRAM_METHODS),
    "spec_fill_dead_bands": ("mel bands above 8 kHz filled", {"spec_dead_band_hz": 8000}, SPECTROGRAM_METHODS),
    "spec_both": ("band z-score + dead bands filled", {"spec_band_norm": "zscore", "spec_dead_band_hz": 8000}, SPECTROGRAM_METHODS),
    "agg_mean": ("score = mean of the anomaly map", {"dcase2026": {"score_aggregation": "mean"}}, None),
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
    if epochs is not None:
        config["epochs"] = epochs  # an experiment may override it below
    # the aggregation is a default an experiment may change; the test protocol is fixed
    config["dcase2026"] = deep_merge(config.get("dcase2026", {}), {"score_aggregation": aggregation})
    config = deep_merge(config, overrides)
    config["dcase2026"] = deep_merge(config["dcase2026"], {"test_domain": "source", "run_name": f"grid_{name}"})
    if name != "train_all":
        config["dcase2026"].setdefault("train_domains", "source")
    return config


def methods_for(name: str, methods: list[str]) -> list[str]:
    only = EXPERIMENTS[name][2]
    return [m for m in methods if only is None or m in only]


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
