"""Does the model localise the anomaly where it really is? Remove it, score the result again and listen to it (EnvMix, BEATs).

For a few anomalous EnvMix test clips of different backgrounds, SNR levels and anomaly classes, the anomaly map of PatchCore (memory
1% of the patches) and of Dinomaly (both on BEATs) becomes a time-frequency mask; the masked region of the spectrogram is replaced by a
background estimate (the median magnitude of the same frequency in the frames that are not masked, never louder than the original),
the original phase is kept and the waveform is rebuilt. Then, for every clip and every mask (PatchCore, Dinomaly, and the ground-truth
mask as the best a mask can do):

* SI-SDR of the cleaned audio against the TRUE background (EnvMix keeps it) and the gain over the unprocessed clip;
* the anomaly score of PatchCore and of Dinomaly on the cleaned audio, as a percentile of the scores of the normal test clips: the
  mask of one model is judged by both models, because a model that scores audio cleaned with its own map is judged by construction;
* wav files (original, cleaned by each mask, true background, the anomaly) and a figure (spectrograms, maps, masks).

    python paper_benchmark/anomaly_removal.py --base-config paper_benchmark/config.beats_benchmark.json
    python paper_benchmark/anomaly_removal.py --base-config ... --combos siren:-6 street_music:0 --examples-per-combo 2
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000
N_FFT, WIN_LENGTH, HOP = 512, 400, 160  # the frames of the BEATs filterbank: 25 ms windows every 10 ms
MEL_BINS, MEL_LOW = 128, 20.0
DEFAULT_COMBOS = ("air_conditioner:6", "engine_idling:-6", "street_music:0", "children_playing:6", "siren:0")
#: the files of an example, in the order to listen to them
AFTER_FILES = {"patchcore": "2_after_patchcore.wav", "dinomaly": "3_after_dinomaly.wav", "oracle": "4_after_oracle_mask.wav"}

README = """# Anomaly removal on EnvMix (BEATs): results to listen to and read

What was done: for every example (an EnvMix test clip: a background sound with an anomaly mixed in) the anomaly map of PatchCore (memory
1% of the patches) and of Dinomaly, both on BEATs, was turned into a time-frequency mask; the masked part of the spectrogram was replaced
by an estimate of the background (median magnitude of the same frequency in the frames that are not masked, never louder than the
original), the original phase was kept and the waveform was rebuilt. The "oracle" mask is the ground-truth mask of the anomaly: it is the
best a mask can do, so it separates the limit of the removal method from the error of the models.

## Folders: `examples/<background>_snr<dB>_<anomaly class>_<clip>/`

| File | What it is |
|---|---|
| `1_before.wav` | the clip with the anomaly (the input) |
| `2_after_patchcore.wav` | the anomaly removed with the PatchCore mask |
| `3_after_dinomaly.wav` | the anomaly removed with the Dinomaly mask |
| `4_after_oracle_mask.wav` | the anomaly removed with the ground-truth mask (the best case) |
| `5_true_background.wav` | the background alone (what a perfect removal would give) |
| `6_anomaly_only.wav` | the anomaly alone (what should disappear) |
| `figure.png` | spectrograms: input, true background, ground-truth mask, the two anomaly maps, the three cleaned clips |

All the files of one example share the same volume scale, so they can be compared by ear.

## Tables

* `metrics.csv`: one row per example and mask, with every number.
* `summary.md`: the main ones.
  * `SI-SDR before / after`: similarity (dB) to the TRUE background; higher is better; the gain is `after - before`.
  * `score pct`: where the anomaly score of the clip falls among the scores of the normal test clips (0 = below all of them, 1 = above all
    of them). Before the removal it should be high, after it should be low. It is given for both detectors: a mask is judged by both
    models, because a model judging the audio cleaned with its own map would be judged by construction.
  * `normal?`: whether the score of the cleaned clip is below the 95th percentile of the scores of the normal clips (the clip would be
    called normal).

## How to read it

* Where `oracle` is good and the model masks are not, the localisation of the model is the limit.
* Where the oracle is also poor, the limit is the method of removal (a sound that overlaps the background in frequency cannot be taken
  away without taking some background with it).
* The maps are coarse (patches of about 160 ms x 16 mel bands): expect a partial attenuation, not a clean removal.
"""


# ---------------------------------------------------------------------------------------------- pure signal functions
def _mel(freq_hz: torch.Tensor) -> torch.Tensor:
    return 1127.0 * torch.log(1.0 + freq_hz / 700.0)


def to_stft_grid(mask_fbank: torch.Tensor, n_stft_frames: int) -> torch.Tensor:
    """A ``(frames, 128)`` map of the BEATs filterbank on the ``(n_fft/2 + 1, n_stft_frames)`` grid of the STFT (bilinear).

    The filterbank bins are equally spaced on the Kaldi mel scale between 20 Hz and Nyquist; the frame ``i`` covers the samples
    ``[160 i, 160 i + 400)``, so its centre is the centre of the (centred) STFT frame ``i + 1.25``.
    """
    frames = mask_fbank.shape[0]
    freqs = torch.arange(N_FFT // 2 + 1, dtype=torch.float32) * SAMPLE_RATE / N_FFT
    mel_low, mel_high = _mel(torch.tensor(MEL_LOW)), _mel(torch.tensor(SAMPLE_RATE / 2))
    delta = (mel_high - mel_low) / (MEL_BINS + 1)
    f_index = ((_mel(freqs) - mel_low) / delta - 1.0).clamp(0, MEL_BINS - 1)
    t_index = ((torch.arange(n_stft_frames, dtype=torch.float32) * HOP - WIN_LENGTH / 2) / HOP).clamp(0, frames - 1)
    t0, f0 = t_index.floor().long(), f_index.floor().long()
    t1, f1 = (t0 + 1).clamp(max=frames - 1), (f0 + 1).clamp(max=MEL_BINS - 1)
    wt, wf = (t_index - t0)[:, None], (f_index - f0)[None, :]
    m = mask_fbank.float()
    grid = (m[t0[:, None], f0[None, :]] * (1 - wt) * (1 - wf) + m[t1[:, None], f0[None, :]] * wt * (1 - wf)
            + m[t0[:, None], f1[None, :]] * (1 - wt) * wf + m[t1[:, None], f1[None, :]] * wt * wf)
    return grid.T  # (freq, time)


def maps_to_mask(anomaly_map: torch.Tensor, tau: float, dilate: int = 5, blur: int = 5) -> torch.Tensor:
    """``(frames, 128)`` anomaly map -> mask in [0, 1]: the pixels above ``tau``, grown by ``dilate`` and with soft edges (``blur``)."""
    mask = (anomaly_map.float() > tau).float()[None, None]
    if dilate > 1:
        mask = F.max_pool2d(mask, dilate, stride=1, padding=dilate // 2)
    if blur > 1:
        mask = F.avg_pool2d(mask, blur, stride=1, padding=blur // 2, count_include_pad=False)
    return mask[0, 0].clamp(0, 1)


def remove_anomaly(waveform: torch.Tensor, mask_fbank: torch.Tensor, min_background_frames: int = 5) -> torch.Tensor:
    """The waveform with the masked part of its spectrogram replaced by a background estimate, the phase unchanged.

    ``mask_fbank`` is ``(frames, 128)`` in [0, 1]. For every frequency the background magnitude is the median over the frames where
    the mask is below 0.5 (the 20th percentile of all the frames when there are fewer than ``min_background_frames``); the new
    magnitude is ``(1 - m) |X| + m min(|X|, background)``: the processing can only take energy away.
    """
    window = torch.hann_window(WIN_LENGTH)
    spectrum = torch.stft(waveform, N_FFT, hop_length=HOP, win_length=WIN_LENGTH, window=window, center=True, return_complex=True)
    magnitude, phase = spectrum.abs(), torch.angle(spectrum)
    mask = to_stft_grid(mask_fbank, magnitude.shape[1])
    mask = mask.clamp(0, 1)
    unmasked = mask < 0.5
    background = torch.where(unmasked, magnitude, torch.full_like(magnitude, float("nan"))).nanmedian(dim=1).values
    fallback = torch.quantile(magnitude, 0.2, dim=1)
    background = torch.where(unmasked.sum(dim=1) >= min_background_frames, background, fallback)
    cleaned = magnitude * (1 - mask) + mask * torch.minimum(magnitude, background[:, None])
    return torch.istft(torch.polar(cleaned, phase), N_FFT, hop_length=HOP, win_length=WIN_LENGTH, window=window, center=True,
                       length=waveform.shape[0])


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """Scale-invariant SDR in dB of ``estimate`` against ``reference``."""
    estimate, reference = estimate.double() - estimate.double().mean(), reference.double() - reference.double().mean()
    target = (estimate @ reference) / (reference @ reference).clamp_min(1e-20) * reference
    noise = estimate - target
    return float(10 * torch.log10((target @ target).clamp_min(1e-20) / (noise @ noise).clamp_min(1e-20)))


def percentile_of(score: float, normal_scores: np.ndarray) -> float:
    """The share of the normal test clips whose score is below ``score`` (0 = lower than all of them, 1 = higher than all)."""
    return float((np.asarray(normal_scores) < score).mean())


def pick_examples(categories: dict[int, str], labels: list[int], count: int) -> list[int]:
    """Indices of ``count`` anomalous clips of different anomaly classes (the first of each class, in clip order)."""
    chosen, seen = [], set()
    for index, label in enumerate(labels):
        category = categories.get(index)
        if label == 1 and category is not None and category not in seen:
            chosen.append(index)
            seen.add(category)
        if len(chosen) == count:
            break
    return chosen


# ---------------------------------------------------------------------------------------------- one clip
class Detector:
    """A fitted model seen through its anomaly map and score, with what is needed to read them (threshold, normal scores)."""

    def __init__(self, name: str, model, device: torch.device, tau: float = 0.0, normal_scores=None):
        self.name, self.model, self.device = name, model, device
        self.tau = tau
        self.normal_scores = np.asarray([] if normal_scores is None else normal_scores)

    @torch.no_grad()
    def map_and_score(self, waveform: torch.Tensor) -> tuple[np.ndarray, float]:
        maps, scores, _ = self.model(waveform.float().unsqueeze(0).to(self.device))
        return maps[0, 0].detach().cpu().numpy(), float(scores[0])


def localisation_metrics(anomaly_map: np.ndarray, truth: np.ndarray, tau: float) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    truth = truth.astype(bool)
    predicted = anomaly_map > tau
    union = (predicted | truth).sum()
    out = {"iou": float((predicted & truth).sum() / union) if union else float("nan"), "predicted_fraction": float(predicted.mean())}
    out["pixel_auc"] = float(roc_auc_score(truth.ravel(), anomaly_map.ravel())) if 0 < truth.sum() < truth.size else float("nan")
    return out


def process_example(detectors: dict[str, Detector], mixture: torch.Tensor, background: torch.Tensor, truth_mask: np.ndarray,
                    label: str, normal_quantile: float = 0.95, save=None) -> list[dict]:
    """One row per mask (``<detector>`` and ``oracle``): SI-SDR against the true background and both detectors' scores on the result.

    ``save(variant, cleaned)`` is called for every cleaned waveform (to write it), if given.
    """
    maps, mixture_scores = {}, {}
    for name, detector in detectors.items():
        maps[name], mixture_scores[name] = detector.map_and_score(mixture)
    masks = {name: maps_to_mask(torch.as_tensor(maps[name]), detectors[name].tau) for name in detectors}
    masks["oracle"] = maps_to_mask(torch.as_tensor(truth_mask).float(), 0.5)
    mixture_sdr = si_sdr(mixture, background)
    background_scores = {name: d.map_and_score(background)[1] for name, d in detectors.items()}
    rows = []
    for variant, mask in masks.items():
        cleaned = remove_anomaly(mixture, mask)
        if save is not None:
            save(variant, cleaned)
        row = {"example": label, "mask": variant, "mask_fraction": float(mask.mean()), "si_sdr_mixture": mixture_sdr,
               "si_sdr_cleaned": si_sdr(cleaned, background), "si_sdr_gain": si_sdr(cleaned, background) - mixture_sdr}
        if variant in detectors:
            row.update({f"loc_{k}": v for k, v in localisation_metrics(maps[variant], truth_mask, detectors[variant].tau).items()})
        for name, detector in detectors.items():
            after = detector.map_and_score(cleaned)[1]
            limit = float(np.quantile(detector.normal_scores, normal_quantile)) if len(detector.normal_scores) else float("inf")
            row.update({f"score_{name}_mixture": mixture_scores[name], f"score_{name}_cleaned": after,
                        f"score_{name}_background": background_scores[name],
                        f"pct_{name}_mixture": percentile_of(mixture_scores[name], detector.normal_scores),
                        f"pct_{name}_cleaned": percentile_of(after, detector.normal_scores),
                        f"normal_{name}_cleaned": bool(after <= limit)})
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------- figure
def save_figure(path: Path, mixture, background, maps: dict[str, np.ndarray], truth: np.ndarray, cleaned: dict[str, torch.Tensor]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from moviad.utilities.audio.beats_feature_extractor import BEATsFbank

    fbank = BEATsFbank()
    spec = lambda w: fbank(w.float().unsqueeze(0))[0, 0].T.numpy()  # noqa: E731  (mel bins x frames)
    panels = [("mixture", spec(mixture)), ("true background", spec(background)), ("ground-truth mask", truth.T)]
    panels += [(f"map: {name}", m.T) for name, m in maps.items()]
    panels += [(f"cleaned ({name} mask)", spec(w)) for name, w in cleaned.items()]
    columns = 4
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.6 * rows))
    for axis in np.atleast_1d(axes).ravel():
        axis.axis("off")
    for axis, (title, image) in zip(np.atleast_1d(axes).ravel(), panels):
        axis.axis("on")
        axis.imshow(image, origin="lower", aspect="auto", cmap="magma")
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_wav(path: Path, waveform: torch.Tensor, scale: float) -> None:
    import torchaudio

    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), (waveform.float() * scale).clamp(-1, 1).unsqueeze(0), SAMPLE_RATE)


# ---------------------------------------------------------------------------------------------- the experiment
def summary_table(rows: list[dict], detectors: list[str]) -> str:
    header = ["example", "mask", "SI-SDR before", "SI-SDR after", "gain (dB)"]
    for name in detectors:
        header += [f"{name}: score pct before", f"{name}: score pct after", f"{name}: normal?"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        cells = [r["example"], r["mask"], f"{r['si_sdr_mixture']:.1f}", f"{r['si_sdr_cleaned']:.1f}", f"{r['si_sdr_gain']:+.1f}"]
        for name in detectors:
            cells += [f"{r[f'pct_{name}_mixture']:.2f}", f"{r[f'pct_{name}_cleaned']:.2f}", "yes" if r[f"normal_{name}_cleaned"] else "no"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_combo(base: dict, background_name: str, snr: float, seed: int, examples: int, output: Path, quantile: float) -> list[dict]:
    from moviad.datasets.audio_dataset import SpectrogramBinarizer
    from benchmark_common import fit_model, make_model, resolve_device, set_seed
    from run_beats_benchmark import build_config
    from run_envmix import build_loaders

    configs = {"patchcore": build_config(base, "envmix", "patchcore_beats_std_1pct", str(output), [seed], [seed]),
               "dinomaly": build_config(base, "envmix", "dinomaly_beats", str(output), [seed], [seed])}
    config = configs["patchcore"]
    device = resolve_device(config.get("device", "auto"))
    set_seed(seed)
    # one dataset for both models: they have to see the same clips
    train_dataset, test_dataset, train_loader, test_loader, _ = build_loaders(config, background_name, snr, seed, False, "patchcore")
    input_size = tuple(int(d) for d in test_dataset.img_shape[1:])
    detectors = {}
    for name, cfg in configs.items():
        model = make_model(name, cfg, device, input_size, class_name=background_name)
        fit_model(name, model, train_loader, test_loader, cfg, device, False)
        model.eval()
        detectors[name] = Detector(name, model, device)
    # thresholds and normal scores from the normal clips of the test set
    labels, normal_maps = [], {name: [] for name in detectors}
    normal_scores = {name: [] for name in detectors}
    for images, label_batch, _masks, _tmp, _ in test_loader:
        for name, detector in detectors.items():
            maps, scores, _ = detector.model(images.float().to(device))
            keep = (label_batch == 0).nonzero().flatten().tolist()
            normal_maps[name] += [maps[i, 0].detach().cpu().numpy() for i in keep]
            normal_scores[name] += [float(scores[i]) for i in keep]
        labels += [int(x) for x in label_batch]
    for name, detector in detectors.items():
        detector.tau = float(np.quantile(np.concatenate([m.ravel() for m in normal_maps[name]]), quantile))
        detector.normal_scores = np.asarray(normal_scores[name])
    underlying = test_dataset.dataset  # the AudioAnomalyDataset: the clips, their true background and the anomaly
    chosen = pick_examples(getattr(underlying, "anomaly_category_mapping", {}), labels, examples)
    rows = []
    for index in chosen:
        sample = underlying.dataset[index]
        energy = np.asarray(test_dataset[index][2]).squeeze()
        truth = SpectrogramBinarizer.by_energy_threshold(0.4, energy)
        category = underlying.anomaly_category_mapping[index]
        label = f"{background_name}_snr{snr:g}_{category}_{index}"
        folder = output / "examples" / label
        scale = 0.9 / float(sample.audio_clip.abs().max().clamp_min(1e-8))
        cleaned_audio = {}

        def save(variant, cleaned, folder=folder, scale=scale):
            cleaned_audio[variant] = cleaned
            save_wav(folder / AFTER_FILES[variant], cleaned, scale)

        rows += process_example(detectors, sample.audio_clip, sample.bg_clip, truth, label, save=save)
        save_wav(folder / "1_before.wav", sample.audio_clip, scale)  # the same scale for every file of the example: the levels are comparable
        save_wav(folder / "5_true_background.wav", sample.bg_clip, scale)
        save_wav(folder / "6_anomaly_only.wav", sample.anomaly_clip, scale)
        maps = {name: d.map_and_score(sample.audio_clip)[0] for name, d in detectors.items()}
        save_figure(folder / "figure.png", sample.audio_clip, sample.bg_clip, maps, truth, cleaned_audio)
    del detectors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper_benchmark/anomaly_removal"))
    parser.add_argument("--combos", nargs="+", default=list(DEFAULT_COMBOS), metavar="BACKGROUND:SNR_DB")
    parser.add_argument("--examples-per-combo", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1, help="not 0: with 0 the contamination of EnvMix is not seeded")
    parser.add_argument("--threshold-quantile", type=float, default=0.99, help="the map threshold: this quantile of the pixels of the normal test clips")
    args = parser.parse_args()
    base = json.loads(args.base_config.expanduser().read_text(encoding="utf-8"))
    all_rows: list[dict] = []
    for combo in args.combos:
        background, snr = combo.rsplit(":", 1)
        print(f"\n===== {background} at {float(snr):g} dB", flush=True)
        all_rows += run_combo(base, background, float(snr), args.seed, args.examples_per_combo, args.output_dir, args.threshold_quantile)
        if all_rows:  # written after every combo: a crash keeps what is done
            args.output_dir.mkdir(parents=True, exist_ok=True)
            fields = list(dict.fromkeys(k for row in all_rows for k in row))
            with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(all_rows)
            (args.output_dir / "summary.md").write_text(summary_table(all_rows, ["patchcore", "dinomaly"]) + "\n", encoding="utf-8")
            (args.output_dir / "README.md").write_text(README, encoding="utf-8")
    if all_rows:
        archive = shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir)  # the folder to download
        print(f"\nResults in {args.output_dir}\nZip to download: {archive}")
    else:
        print("\nNo example was produced")


if __name__ == "__main__":
    main()
