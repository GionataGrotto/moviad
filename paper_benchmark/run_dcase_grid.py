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
    beats_late / beats_mid   BEATs transformer blocks 8,10,12 / 4,6,8 as the backbone of patchcore, padim, cfa
                 (needs beats_checkpoint / beats_code_dir, or the knn_beats_* keys, in the base config)
    beats_* search: block choice (b12, b10, b10_12, b6_9_12), protocol (+ target clips, near - far,
                 mean score) and detector settings (PaDiM full covariance / 576 channels,
                 PatchCore bank 60000, CFA gamma_c=1), all around beats_late. Presets:
                 beats_layers, beats_protocol, beats_heads, beats_search (all of them), band_search, band_full
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
# heads that run on feature maps; STFPM trains its student with an SGD setup made for CNNs, so it is left out
BEATS_METHODS = {"patchcore", "padim", "cfa"}
BEATS_LATE = {"backbone": "BEATs", "layers": [8, 10, 12]}
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
    "beats_late": ("BEATs blocks 8,10,12 as backbone", {"backbone": "BEATs", "layers": [8, 10, 12]}, BEATS_METHODS),
    "beats_mid": ("BEATs blocks 4,6,8 as backbone", {"backbone": "BEATs", "layers": [4, 6, 8]}, BEATS_METHODS),
    # ---- search for the best BEATs configuration, one factor at a time around beats_late ----
    "beats_b12": ("BEATs block 12 only", {"backbone": "BEATs", "layers": [12]}, BEATS_METHODS),
    "beats_b10": ("BEATs block 10 only", {"backbone": "BEATs", "layers": [10]}, BEATS_METHODS),
    "beats_b10_12": ("BEATs blocks 10,11,12", {"backbone": "BEATs", "layers": [10, 11, 12]}, BEATS_METHODS),
    "beats_b6_9_12": ("BEATs blocks 6,9,12", {"backbone": "BEATs", "layers": [6, 9, 12]}, BEATS_METHODS),
    "beats_late_train_all": ("beats_late + 10 target clips", {**BEATS_LATE, "dcase2026": {"train_domains": "all"}}, BEATS_METHODS),
    "beats_late_diff": ("beats_late on near - far", {**BEATS_LATE, "dcase2026": {"channel": "diff"}}, BEATS_METHODS),
    "beats_late_agg_mean": ("beats_late, score = mean of the map", {**BEATS_LATE, "dcase2026": {"score_aggregation": "mean"}}, BEATS_METHODS),
    "beats_padim_full_cov": ("beats_late, PaDiM full covariance", {**BEATS_LATE, "padim_diag_cov": False}, {"padim"}),
    "beats_padim_d576": ("beats_late, PaDiM 576 channels (25%)", {**BEATS_LATE, "padim_embedding_dim": 576}, {"padim"}),
    "beats_pc_bank60k": ("beats_late, PatchCore bank 60000", {**BEATS_LATE, "memory_bank_size": 60000}, {"patchcore"}),
    "beats_cfa_gamma1": ("beats_late, CFA gamma_c=1", {**BEATS_LATE, "cfa_gamma_c": 1}, {"cfa"}),
    # ---- per-band ideas on top of the best BEATs setups: see the "band_search" preset ----
    "band_pc_banks": ("PatchCore, one memory bank per band", {**BEATS_LATE, "patchcore_band_banks": True}, {"patchcore"}),
    "band_pc_std": ("PatchCore, per-band score standardisation", {**BEATS_LATE, "patchcore_score_std": "band"}, {"patchcore"}),
    "band_pc_banks_std": ("PatchCore, per-band banks + score standardisation",
                          {**BEATS_LATE, "patchcore_band_banks": True, "patchcore_score_std": "band"}, {"patchcore"}),
    "band_padim_gauss": ("PaDiM, one diagonal Gaussian per band", {**BEATS_LATE, "padim_per_band": True}, {"padim"}),
    "band_padim_gauss_full": ("PaDiM, one full-covariance Gaussian per band",
                              {**BEATS_LATE, "padim_per_band": True, "padim_diag_cov": False}, {"padim"}),
    "band_padim_gauss_full_std": ("PaDiM, per-band full Gaussians + score standardisation",
                                  {**BEATS_LATE, "padim_per_band": True, "padim_diag_cov": False, "padim_score_std": "band"}, {"padim"}),
    # ---- STFPM: optimiser and BEATs student (the trainer used to hard-code SGD with lr 0.4) ----
    "stfpm_sgd_lr0p1": ("STFPM, SGD lr 0.1", {"stfpm_lr": 0.1}, {"stfpm"}),
    "stfpm_sgd_lr0p02": ("STFPM, SGD lr 0.02", {"stfpm_lr": 0.02}, {"stfpm"}),
    "stfpm_adamw_1e4": ("STFPM, AdamW lr 1e-4", {"stfpm_optimizer": "adamw", "stfpm_lr": 1e-4}, {"stfpm"}),
    "stfpm_beats_late_adamw_1e5": ("STFPM on BEATs 8,10,12, AdamW lr 1e-5",
                                   {**BEATS_LATE, "stfpm_optimizer": "adamw", "stfpm_lr": 1e-5}, {"stfpm"}),
    "stfpm_beats_late_adamw_1e4": ("STFPM on BEATs 8,10,12, AdamW lr 1e-4",
                                   {**BEATS_LATE, "stfpm_optimizer": "adamw", "stfpm_lr": 1e-4}, {"stfpm"}),
    "stfpm_beats_b12_adamw_1e5": ("STFPM on BEATs block 12, AdamW lr 1e-5",
                                  {"backbone": "BEATs", "layers": [12], "stfpm_optimizer": "adamw", "stfpm_lr": 1e-5}, {"stfpm"}),
    # ---- far-microphone least-squares denoising (risky on the *Emu machines, see ls_denoise) ----
    "beats_late_ls": ("beats_late on near - LS(far)", {**BEATS_LATE, "dcase2026": {"channel": "ls"}}, BEATS_METHODS),
    "beats_late_ls_half": ("beats_late on near - 0.5 LS(far)", {**BEATS_LATE, "dcase2026": {"channel": "ls_half"}}, BEATS_METHODS),
    "knn_ls": ("kNN BEATs on near - LS(far)", {"dcase2026": {"channel": "ls"}}, {"knn"}),
    # ---- other detectors on the BEATs / signal side ----
    "handcrafted_ls": ("handcrafted on near - LS(far)", {"dcase2026": {"channel": "ls"}}, {"handcrafted"}),
    "handcrafted_64_ls": ("64 sub-bands on near - LS(far)", {"handcrafted_bands": 64, "dcase2026": {"channel": "ls"}}, {"handcrafted"}),
    "handcrafted_128": ("128 sub-bands + shrunk Mahalanobis", {"handcrafted_bands": 128}, {"handcrafted"}),
    "beats_b12_ls": ("BEATs block 12 on near - LS(far)", {"backbone": "BEATs", "layers": [12], "dcase2026": {"channel": "ls"}}, BEATS_METHODS),
    "band_pc_banks_std_ls": ("PatchCore per-band banks + std on near - LS(far)",
                             {**BEATS_LATE, "patchcore_band_banks": True, "patchcore_score_std": "band", "dcase2026": {"channel": "ls"}}, {"patchcore"}),
    "beats_padim_full_cov_ls": ("PaDiM full covariance on near - LS(far)",
                                {**BEATS_LATE, "padim_diag_cov": False, "dcase2026": {"channel": "ls"}}, {"padim"}),
    "knn_b12_ls": ("kNN block 12 on near - LS(far)", {"knn_layers": [12], "dcase2026": {"channel": "ls"}}, {"knn"}),
    "knn_ft_ls": ("LoRA + ArcFace kNN on near - LS(far)", {"dcase2026": {"channel": "ls"}}, {"knn_ft"}),
    "knn_ft_b12_ls": ("LoRA + ArcFace kNN block 12 on near - LS(far)", {"knn_layers": [12], "dcase2026": {"channel": "ls"}}, {"knn_ft"}),
    "handcrafted_64_ls_shrink0p3": ("64 sub-bands on LS, shrinkage 0.3", {"handcrafted_bands": 64, "handcrafted_shrinkage": 0.3, "dcase2026": {"channel": "ls"}}, {"handcrafted"}),
    "handcrafted_64_ls_shrink0p03": ("64 sub-bands on LS, shrinkage 0.03", {"handcrafted_bands": 64, "handcrafted_shrinkage": 0.03, "dcase2026": {"channel": "ls"}}, {"handcrafted"}),
    "handcrafted_96_ls": ("96 sub-bands on LS", {"handcrafted_bands": 96, "dcase2026": {"channel": "ls"}}, {"handcrafted"}),
    # ---- round 4: finer LS, the LS on the other methods, richer handcrafted features, stronger fine-tuning ----
    "band_pc_banks_std_ls_win": ("PatchCore banks + std, LS on 2 s blocks", {**BEATS_LATE, "patchcore_band_banks": True, "patchcore_score_std": "band", "dcase2026": {"channel": "ls_win"}}, {"patchcore"}),
    "band_pc_banks_std_ls_global": ("PatchCore banks + std, one LS filter per machine", {**BEATS_LATE, "patchcore_band_banks": True, "patchcore_score_std": "band", "dcase2026": {"channel": "ls_global"}}, {"patchcore"}),
    "handcrafted_64_ls_win": ("64 sub-bands, LS on 2 s blocks", {"handcrafted_bands": 64, "dcase2026": {"channel": "ls_win"}}, {"handcrafted"}),
    "handcrafted_64_ls_global": ("64 sub-bands, one LS filter per machine", {"handcrafted_bands": 64, "dcase2026": {"channel": "ls_global"}}, {"handcrafted"}),
    "knn_ls_win": ("kNN BEATs, LS on 2 s blocks", {"dcase2026": {"channel": "ls_win"}}, {"knn"}),
    "knn_ls_global": ("kNN BEATs, one LS filter per machine", {"dcase2026": {"channel": "ls_global"}}, {"knn"}),
    "stfpm_ls": ("STFPM (Cnn14) on near - LS(far)", {"dcase2026": {"channel": "ls"}}, {"stfpm"}),
    "dinomaly_ls": ("Dinomaly on near - LS(far)", {"dcase2026": {"channel": "ls"}}, {"dinomaly"}),
    "handcrafted_64_ls_kurt": ("64 sub-bands on LS, + kurtosis", {"handcrafted_bands": 64, "handcrafted_stats": ["mean", "std", "diffstd", "kurtosis"], "dcase2026": {"channel": "ls"}}, {"handcrafted"}),
    "handcrafted_64_ls_quant": ("64 sub-bands on LS, + quantiles 10/90", {"handcrafted_bands": 64, "handcrafted_stats": ["mean", "std", "diffstd", "q10", "q90"], "dcase2026": {"channel": "ls"}}, {"handcrafted"}),
    "handcrafted_64_ls_corr": ("64 sub-bands on LS, + neighbour-band correlation", {"handcrafted_bands": 64, "handcrafted_stats": ["mean", "std", "diffstd", "corr"], "dcase2026": {"channel": "ls"}}, {"handcrafted"}),
    "handcrafted_64_ls_allstats": ("64 sub-bands on LS, all statistics", {"handcrafted_bands": 64, "handcrafted_stats": ["mean", "std", "diffstd", "kurtosis", "q10", "q90", "corr"], "dcase2026": {"channel": "ls"}}, {"handcrafted"}),
    "handcrafted_64_ls_knn": ("64 sub-bands on LS, kNN scorer", {"handcrafted_bands": 64, "handcrafted_scorer": "knn", "dcase2026": {"channel": "ls"}}, {"handcrafted"}),
    "handcrafted_64_ls_iforest": ("64 sub-bands on LS, isolation forest", {"handcrafted_bands": 64, "handcrafted_scorer": "iforest", "dcase2026": {"channel": "ls"}}, {"handcrafted"}),
    "knn_ft_ls_strong": ("LoRA + ArcFace on LS, 30 epochs, rank 16, margin 0.5", {"finetune_epochs": 30, "finetune_rank": 16, "finetune_margin": 0.5, "dcase2026": {"channel": "ls"}}, {"knn_ft"}),
    "knn_ft_ls_pseudo": ("LoRA + ArcFace on LS, k-means pseudo-classes without attributes", {"finetune_pseudo_classes": 4, "dcase2026": {"channel": "ls"}}, {"knn_ft"}),
    "knn_b10_12_ls": ("kNN blocks 10 and 12 on LS", {"knn_layers": [10, 12], "dcase2026": {"channel": "ls"}}, {"knn"}),
    "knn_b8_10_12_ls": ("kNN blocks 8, 10, 12 on LS", {"knn_layers": [8, 10, 12], "dcase2026": {"channel": "ls"}}, {"knn"}),

    "knn_b12": ("kNN on BEATs block 12", {"knn_layers": [12]}, {"knn"}),
    "handcrafted": ("sub-band statistics + shrunk Mahalanobis", {}, {"handcrafted"}),
    "handcrafted_64": ("64 sub-bands + shrunk Mahalanobis", {"handcrafted_bands": 64}, {"handcrafted"}),
    "knn_ft": ("kNN on BEATs fine-tuned with LoRA + ArcFace", {}, {"knn_ft"}),
    "knn_ft_b12": ("LoRA + ArcFace, kNN on block 12", {"knn_layers": [12]}, {"knn_ft"}),
    "dinomaly_l2": ("Dinomaly, L2 distance instead of cosine", {"dinomaly_distance": "l2"}, {"dinomaly"}),
    "agg_mean": ("score = mean of the anomaly map", {"dcase2026": {"score_aggregation": "mean"}}, None),
}
DEFAULT_METHODS = ["patchcore", "padim", "cfa", "stfpm", "dinomaly"]
#: ready-made groups, run in this order: first which blocks, then the protocol, then the detector settings
PRESETS = {
    "beats_layers": ["beats_late", "beats_mid", "beats_b12", "beats_b10", "beats_b10_12", "beats_b6_9_12"],
    "beats_protocol": ["beats_late_train_all", "beats_late_diff", "beats_late_agg_mean"],
    "beats_heads": ["beats_padim_full_cov", "beats_padim_d576", "beats_pc_bank60k", "beats_cfa_gamma1"],
}
PRESETS["band_search"] = [
    "band_pc_banks", "band_pc_std", "band_pc_banks_std",
    "band_padim_gauss", "band_padim_gauss_full", "band_padim_gauss_full_std",
]
#: the band ideas next to their references (beats_late for PatchCore, full covariance for PaDiM), rerun with the same code
#: STFPM: optimiser and layers on Cnn14, then a BEATs student; use with --methods stfpm
PRESETS["stfpm_search"] = [
    "baseline", "stfpm_sgd_lr0p1", "stfpm_sgd_lr0p02", "stfpm_adamw_1e4", "layers_fine", "layers_deep",
    "stfpm_beats_late_adamw_1e5", "stfpm_beats_late_adamw_1e4", "stfpm_beats_b12_adamw_1e5",
]
PRESETS["ls_search"] = ["beats_late_ls", "beats_late_ls_half", "knn_ls"]
PRESETS["signal_search"] = ["handcrafted", "handcrafted_64", "knn_b12"]
PRESETS["finetune_search"] = ["knn_ft", "knn_ft_b12"]
PRESETS["band_full"] = ["beats_late", "beats_padim_full_cov"] + PRESETS["band_search"]
#: aggregations compared from the same trained model by --aggregation-sweep
AGGREGATION_SWEEP = ["max", "mean", "top5pct", "top10pct"]
PRESETS["beats_search"] = PRESETS["beats_layers"] + PRESETS["beats_protocol"] + PRESETS["beats_heads"]


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config(base: dict, name: str, epochs: int | None, aggregation: str = "temporal_topk_mean", suffix: str = "",
                 extra_aggregations: tuple[str, ...] = ()) -> dict:
    description, overrides, _ = EXPERIMENTS[name]
    config = deep_merge(base, {"backbone": "Cnn14", "layers": LAYERS_BLOCKS_2_4})
    if epochs is not None:
        config["epochs"] = epochs  # an experiment may override it below
    # the aggregation is a default an experiment may change; the test protocol is fixed
    config["dcase2026"] = deep_merge(config.get("dcase2026", {}), {"score_aggregation": aggregation})
    config = deep_merge(config, overrides)
    config["dcase2026"] = deep_merge(config["dcase2026"], {"test_domain": "source", "run_name": f"grid_{name}{suffix}"})
    if extra_aggregations:
        config["dcase2026"]["extra_aggregations"] = [a for a in extra_aggregations if a != config["dcase2026"]["score_aggregation"]]
    if name != "train_all":
        config["dcase2026"].setdefault("train_domains", "source")
    return config


def summary_labels(methods: list[str], extra_aggregations: tuple[str, ...] = (), primary: str = "temporal_topk_mean") -> list[str]:
    """Summary columns: every method, then its extra aggregations written as ``method@aggregation``."""
    labels = list(methods)
    for method in methods:
        labels += [f"{method}@{a}" for a in extra_aggregations if a != primary]
    return labels


def methods_for(name: str, methods: list[str]) -> list[str]:
    only = EXPERIMENTS[name][2]
    return [m for m in methods if only is None or m in only]


def collect(experiments: list[str], methods: list[str], domain: str = "source", suffix: str = "", root: Path = ROOT) -> dict:
    table = {}
    for name in experiments:
        run = root / "dcase2026_task2_evaluator" / "dev" / f"grid_{name}{suffix}" / "teams" / "moviad"
        for method in methods:
            folder = run / method
            results = evaluate_folder(folder, domain) if folder.exists() else {}
            aucs = [auc for auc, _, _ in results.values() if auc is not None]
            table[(name, method)] = (sum(aucs) / len(aucs), len(aucs)) if aucs else None
    return table


MACHINES = ["ToyCar", "ToyCarEmu", "bearingEmu", "fan", "gearboxEmu", "sliderEmu", "valveEmu"]


def machine_detail(experiments: list[str], methods: list[str], suffix: str = "", domain: str = "source", root: Path = ROOT) -> list[str]:
    """Markdown tables with the AUC of every machine for every run that has score files."""
    lines = []
    for name in experiments:
        run = root / "dcase2026_task2_evaluator" / "dev" / f"grid_{name}{suffix}" / "teams" / "moviad"
        block = []
        for method in methods:
            folder = run / method
            results = evaluate_folder(folder, domain) if folder.exists() else {}
            if not results:
                continue
            cells = [f"{results[m][0]:.3f}" if m in results and results[m][0] is not None else "-" for m in MACHINES]
            aucs = [results[m][0] for m in MACHINES if m in results and results[m][0] is not None]
            paucs = [results[m][1] for m in MACHINES if m in results and results[m][1] is not None]
            block.append(f"| {method} | " + " | ".join(cells) + f" | {sum(aucs) / len(aucs):.3f} | {sum(paucs) / len(paucs):.3f} |")
        if block:
            lines += ["", f"### {name}: {EXPERIMENTS[name][0]}", "",
                      "| Metodo | " + " | ".join(MACHINES) + " | Media AUC | Media pAUC |", "|---|" + "---|" * (len(MACHINES) + 2)] + block
    return lines


def pending_methods(name: str, methods: list[str], suffix: str = "", root: Path = ROOT) -> list[str]:
    """The methods of an experiment that do not yet have score files for all seven machines."""
    table = collect([name], methods, suffix=suffix, root=root)
    return [m for m in methods if table[(name, m)] is None or table[(name, m)][1] < len(MACHINES)]


def best_per_method(table: dict, experiments: list[str], methods: list[str]) -> dict:
    """The experiment with the highest mean AUC for every method (only runs that finished all 7 machines)."""
    best = {}
    for method in methods:
        complete = [(table[(name, method)][0], name) for name in experiments
                    if table.get((name, method)) is not None and table[(name, method)][1] == 7]
        if complete:
            value, name = max(complete)
            best[method] = (name, value)
    return best


def write_summary(path: Path, experiments: list[str], methods: list[str], suffix: str = "", root: Path = ROOT) -> None:
    table = collect(experiments, methods, suffix=suffix, root=root)
    lines = ["# Griglia DCASE 2026: AUC source media (7 macchine)", "",
             "| Esperimento | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for name in experiments:
        only = EXPERIMENTS[name][2]
        cells = []
        for method in methods:
            if only is not None and method.split("@")[0] not in only:
                cells.append("-")  # the experiment does not apply to this method
                continue
            value = table[(name, method)]
            cells.append("n/d" if value is None else f"{value[0]:.3f}" + ("" if value[1] == 7 else f" ({value[1]} macchine)"))
        lines.append(f"| {name}: {EXPERIMENTS[name][0]} | " + " | ".join(cells) + " |")
    lines += ["", "`-` = l'esperimento non si applica al metodo; `n/d` = applicabile ma senza risultati."]
    best = best_per_method(table, experiments, methods)
    if best:
        lines += ["", "Migliore per metodo (solo run completi su 7 macchine):", ""]
        lines += [f"- {method}: {name} ({value:.3f})" for method, (name, value) in best.items()]
    detail = machine_detail(experiments, methods, suffix, root=root)
    if detail:
        lines += ["", "## Dettaglio per macchina (AUC source)"] + detail
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", type=Path, required=True, help="config with dcase2026.dataset_path and the device")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=None)
    parser.add_argument("--preset", choices=list(PRESETS), default=None, help="a ready-made group of experiments")
    parser.add_argument("--epochs", type=int, default=30, help="epochs of cfa/stfpm/dinomaly (patchcore and padim ignore it)")
    parser.add_argument("--aggregation", choices=("max", "mean", "temporal_topk_mean"), default="temporal_topk_mean",
                        help="file-level score; the same for every experiment")
    parser.add_argument("--seed", type=int, default=None, help="seed of the runs (default: the runner's default)")
    parser.add_argument("--run-suffix", default="", help="appended to every run name, e.g. _seed2, so repeated runs do not overwrite each other")
    parser.add_argument("--summary", type=Path, default=None, help="default: results/paper_benchmark/grid_summary<suffix>.md")
    parser.add_argument("--extra-aggregations", nargs="+", default=[],
                        help="more file scores computed from the same trained model (max, mean, top5pct, ...), "
                             "reported as extra columns method@aggregation")
    parser.add_argument("--aggregation-sweep", action="store_true",
                        help=f"shorthand for --extra-aggregations {' '.join(AGGREGATION_SWEEP)}")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip the (experiment, method) pairs that already have score files for all 7 machines")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    if args.preset and args.experiments:
        parser.error("use either --preset or --experiments")
    args.experiments = PRESETS[args.preset] if args.preset else (args.experiments or list(EXPERIMENTS))
    suffix = args.run_suffix
    extras = tuple(dict.fromkeys(args.extra_aggregations + (AGGREGATION_SWEEP if args.aggregation_sweep else [])))
    columns = summary_labels(args.methods, extras, args.aggregation)
    summary_path = args.summary or ROOT / "results" / "paper_benchmark" / f"grid_summary{suffix}.md"
    base = json.loads(args.base_config.expanduser().read_text(encoding="utf-8"))
    config_dir = ROOT / "paper_benchmark" / "grid_configs"
    config_dir.mkdir(exist_ok=True)

    failed = []
    for name in args.experiments:
        methods = methods_for(name, args.methods)
        if args.skip_done:
            pending = pending_methods(name, methods, suffix)
            if pending != methods:
                print(f"--skip-done: {name} already complete for {[m for m in methods if m not in pending]}", flush=True)
            methods = pending
        if not methods:
            continue
        config_path = config_dir / f"config.grid_{name}{suffix}.json"
        config_path.write_text(json.dumps(build_config(base, name, args.epochs, args.aggregation, suffix, extras), indent=2), encoding="utf-8")
        command = [sys.executable, str(HERE / "run_dcase2026_task2.py"), "--config", str(config_path), "--methods", *methods]
        if args.seed is not None:
            command += ["--seed", str(args.seed)]
        print(f"\n=== {name}: {EXPERIMENTS[name][0]}\n$ {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            print(f"!! experiment {name} failed (exit code {result.returncode}); continuing", file=sys.stderr)
            failed.append(name)
        write_summary(summary_path, args.experiments, columns, suffix)
        print(f"summary updated: {summary_path}", flush=True)
    if failed:
        sys.exit(1)  # lets a caller (run_all.py) see that something failed


if __name__ == "__main__":
    main()
