"""MIMII and EnvMix with BEATs: the paper's four methods plus Dinomaly, one command, failures tolerated, resumable.

The old Cnn14 results (``results/paper_benchmark/mimii_results.csv`` and ``envmix_results.csv``) are not repeated: only the BEATs
variants are run, into ``--output-dir`` (``results/paper_benchmark/beats``), and ``build_benchmark_tables.py`` puts the two side by side.

Only what applies to these datasets is used: BEATs as the feature extractor, and for PatchCore the per-band banks with the per-band
score standardisation. The far microphone (LS) and the target-domain clips exist only in DCASE 2026, so they are not used here.

    python paper_benchmark/run_beats_benchmark.py --base-config paper_benchmark/config.knn.json     # everything
    python paper_benchmark/run_beats_benchmark.py --base-config ... --datasets mimii --only patchcore padim
    python paper_benchmark/run_beats_benchmark.py --base-config ... --debug                          # a tiny run first

The base config must have the ``mimii`` and ``envmix`` sections (dataset paths) and the BEATs checkpoint (``beats_checkpoint`` and
``beats_code_dir``, or the ``knn_beats_*`` keys). Each run appends to the CSV as soon as it is finished and ``--skip-done`` is always
on, so launching the command again continues where it stopped.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

BEATS_LATE = {"backbone": "BEATs", "layers": [8, 10, 12]}
#: variant -> (method, config overrides). The cheapest and safest first: a crash costs the least.
VARIANTS = {
    "patchcore_beats_std": ("patchcore", {**BEATS_LATE, "patchcore_band_banks": True, "patchcore_score_std": "band"}),
    # the same PatchCore with a memory of 1% of the training patches, as in the original PatchCore (instead of 30000 patches)
    "patchcore_beats_std_1pct": ("patchcore", {**BEATS_LATE, "patchcore_band_banks": True, "patchcore_score_std": "band", "patchcore_memory_fraction": 0.01}),
    "padim_beats": ("padim", {**BEATS_LATE, "padim_diag_cov": False}),
    "dinomaly_beats": ("dinomaly", {"backbone": "BEATs", "layers": [3, 4, 5, 6, 7, 8, 9, 10], "epochs": 12}),
    "cfa_beats": ("cfa", {"backbone": "BEATs", "layers": [12], "epochs": 30}),
    "stfpm_beats": ("stfpm", {**BEATS_LATE, "stfpm_optimizer": "adamw", "stfpm_lr": 1e-5, "epochs": 30}),  # failed on some DCASE machines
}
RUNNERS = {"mimii": "run_mimii.py", "envmix": "run_envmix.py"}
RESULT_FILES = {"mimii": "mimii_results.csv", "envmix": "envmix_results.csv"}


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        merged[key] = deep_merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def build_config(base: dict, dataset: str, variant: str, output_dir: str, mimii_seeds: list[int], envmix_seeds: list[int], debug: bool = False) -> dict:
    """The config of one (dataset, variant): BEATs, its own output folder, no faithfulness (it needs the Cnn14 frontend)."""
    method, overrides = VARIANTS[variant]
    config = deep_merge(base, overrides)
    config.update({"variant": variant, "output_dir": output_dir, "methods": [method], "faithfulness": {"enabled": False}})
    if debug:  # two training batches only: every other one is held out for the score statistics (the default is one in ten)
        config["score_std_holdout"] = 0.5
    if "mimii" in config:
        config["mimii"] = {**config["mimii"], "seeds": mimii_seeds}
    if "envmix" in config:
        config["envmix"] = {**config["envmix"], "seeds": envmix_seeds}
    return config


def plan(datasets: list[str], only: list[str] | None) -> list[tuple[str, str]]:
    names = [name for name in VARIANTS if only is None or name in only or VARIANTS[name][0] in only]
    return [(dataset, name) for name in names for dataset in datasets]


def run(command: list[str], retries: int, wait: float) -> bool:
    for attempt in range(retries + 1):
        print(f"$ {' '.join(command)}  (attempt {attempt + 1}/{retries + 1})", flush=True)
        if subprocess.run(command, cwd=ROOT).returncode == 0:
            return True
        time.sleep(wait)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=list(RUNNERS), default=list(RUNNERS))
    parser.add_argument("--only", nargs="+", default=None, help="variants (e.g. patchcore_beats_std) or methods (patchcore, padim, ...)")
    parser.add_argument("--output-dir", default="results/paper_benchmark/beats")
    parser.add_argument("--mimii-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--envmix-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-wait", type=float, default=30.0)
    parser.add_argument("--debug", action="store_true", help="a tiny subset of every dataset (the runners' --debug)")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    args = parser.parse_args()

    entries = plan(args.datasets, args.only)
    if args.list:
        print("\n".join(f"{dataset}:{variant}" for dataset, variant in entries))
        return
    base = json.loads(args.base_config.expanduser().read_text(encoding="utf-8"))
    config_dir = HERE / "grid_configs"
    config_dir.mkdir(exist_ok=True)
    output_dir = args.output_dir if not args.debug else args.output_dir + "_debug"

    failed = []
    for index, (dataset, variant) in enumerate(entries, 1):
        print(f"\n===== {index}/{len(entries)}: {dataset}:{variant}", flush=True)
        config = build_config(base, dataset, variant, output_dir, args.mimii_seeds, args.envmix_seeds, debug=args.debug)
        path = config_dir / f"config.beats_{dataset}_{variant}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        method = VARIANTS[variant][0]
        command = [sys.executable, str(HERE / RUNNERS[dataset]), "--config", str(path), "--methods", method, "--skip-done"]
        if args.debug:
            command.append("--debug")
        if not run(command, args.retries, args.retry_wait):
            failed.append(f"{dataset}:{variant}")
    print(f"\nFailed ({len(failed)}): {failed or 'none'}")
    print(f"Results: {ROOT / output_dir}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
