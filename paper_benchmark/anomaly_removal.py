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
DEFAULT_COMBOS = ("air_conditioner:0", "engine_idling:-6", "street_music:0", "children_playing:6", "siren:-6")
#: the files of an example, in the order to listen to them
AFTER_FILES = {"patchcore": "2_after_patchcore.wav", "dinomaly": "3_after_dinomaly.wav", "oracle": "4_after_oracle_mask.wav",
               "patchcore+fine": "7_after_patchcore_fine_mask.wav", "dinomaly+fine": "8_after_dinomaly_fine_mask.wav",
               "fused": "9_after_both_models_mask.wav", "patchcore+iter": "10_after_patchcore_iterative.wav",
               "dinomaly+iter": "11_after_dinomaly_iterative.wav", "audiosep": "12_after_audiosep.wav"}
FILLS = ("cap", "replace", "sample", "nmf")
#: the optional families of methods of process_example
EXTRAS = ("adaptive", "fine", "fusion", "iterative")

README = """# Anomaly removal on EnvMix (BEATs): results to listen to and read

What was done: for every example (an EnvMix test clip: a background sound with an anomaly mixed in) the anomaly map of PatchCore (memory
1% of the patches) and of Dinomaly, both on BEATs, was turned into a time-frequency mask; the masked part of the spectrogram was filled
with an estimate of the background (median magnitude of the same frequency in the frames that are not masked), the original phase was
kept and the waveform was rebuilt. Every model is tried with several thresholds of its map (the 90th, 95th and 99th percentile of the
pixels of the normal test clips: `q0.9`, `q0.95`, `q0.99`), and with `area`: the threshold that gives a mask as big as the true one,
which separates how well the model localises from the choice of the threshold. The `oracle` mask is the ground-truth mask: the best a
mask can do, so it separates the limit of the removal method from the error of the models. Three ways of filling: `sample` (every masked cell takes the
value, with its phase, of a random unmasked frame at the same frequency: real background), `replace` (the masked cells get the
median background level) and `cap` (they are only lowered to it, never raised).

## Methods

Every method gets the same checks (the table columns below). `threshold` says which one:

* `q0.9`, `q0.95`, `q0.99`, `area`: a fixed threshold of the map, or the one that gives a mask as big as the true one.
* `rel0.5`: a threshold that follows the clip (halfway between the lowest and the highest value of its map).
* `q0.95+fine`: the mask of `q0.95` restricted to the cells of the STFT that really stick out of the background (3 dB above the median
  of their frequency): the map says where, the energy of the clip says what.
* `and`, `avg` (mask `fused`): the maps of the two models in units of their own thresholds, combined with the minimum (both must flag
  a pixel) or the mean.
* `q0.95x3`: remove, compute the map again, remove what is still flagged (at most 3 rounds, none if the clip already scores as normal).
* `audiosep` (only when it is installed): the source that AudioSep separates for the text of the anomaly class, subtracted from the clip.
* Fillings: `sample`, `replace`, `cap` and `nmf` (the magnitude of the masked cells is predicted by a low-rank model fitted on the rest
  of the clip, the phase is the one of `sample`).

`control error (dB)` is a check on the filling and the mask, not on the model: the same procedure is applied to the true background alone
(without the anomaly); it should leave it as it is, so the lower the better. Likewise `pct_<model>_control` in `metrics.csv` is where the
score of that clip falls after the procedure: if it rises, the procedure itself makes the clip look anomalous.

## Folders: `examples/<background>_snr<dB>_<anomaly class>_<clip>/`

| File | What it is |
|---|---|
| `1_before.wav` | the clip with the anomaly (the input) |
| `2_after_patchcore.wav` | the anomaly removed with the PatchCore mask (main threshold and filling) |
| `3_after_dinomaly.wav` | the anomaly removed with the Dinomaly mask (main threshold and filling) |
| `4_after_oracle_mask.wav` | the anomaly removed with the ground-truth mask (the best case) |
| `5_true_background.wav` | the background that is inside the clip (what a perfect removal would give) |
| `6_anomaly_only.wav` | the anomaly alone (what should disappear) |
| `7_` ... `12_after_*.wav` | the same for the fine mask, both models together, the iterative removal and AudioSep (when run) |
| `figure.png` | spectrograms: input, background, ground-truth mask, the two anomaly maps, the three cleaned clips |

All the files of one example share the same volume scale, so they can be compared by ear.

## Tables

* `metrics.csv`: one row per example, mask, threshold and filling, with every number.
* `summary.md`: the main threshold and filling, one row per example.
* `aggregate.md`: the mean over the examples of every mask, threshold and filling.

How to read the columns:

* `anomaly removed (dB)`: how much the anomaly was attenuated (0 = untouched; 6 dB = half of its amplitude; 12 dB = a quarter).
* `background error (dB)`: how much the rest of the clip changed, relative to the background (lower = less damage).
* `mask area`: the share of the spectrogram that was masked.
* `pixel AUC`, `IoU`: how well the map (or its mask) overlaps the true anomaly.
* `score pct`: where the anomaly score of the clip falls among the scores of the normal test clips (0 = below all of them, 1 = above all
  of them). Before the removal it should be high, after it should be low. It is given for both detectors: a mask is judged by both
  models, because a model judging the audio cleaned with its own map would be judged by construction. `normal?`: the score is below the
  95th percentile of the normal clips (the clip would be called normal).

## How to read it

* Where `oracle` removes the anomaly and the model masks do not, the localisation of the model is the limit; `area` tells if it is the
  threshold (the `area` mask removes much more than the thresholded ones) or the position of the map (it does not either).
* Where the oracle is also poor, the limit is the method of removal (a sound that overlaps the background in frequency cannot be taken
  away without taking some background with it).
* The maps are coarse (patches of about 160 ms x 16 mel bands): a short event may be below what they resolve.
* The background that EnvMix stores as `bg_clip` is not the one inside the clip (it is louder by a constant factor, because the mix uses
  an amplitude ratio and the level factor a power ratio): the reference here is the clip minus the anomaly.
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


def stft_frames(n_samples: int) -> int:
    """The number of frames of the (centred) STFT of ``n_samples`` samples."""
    return n_samples // HOP + 1


def _stft(waveform: torch.Tensor):
    window = torch.hann_window(WIN_LENGTH)
    spectrum = torch.stft(waveform, N_FFT, hop_length=HOP, win_length=WIN_LENGTH, window=window, center=True, return_complex=True)
    return spectrum, window


def background_magnitude(magnitude: torch.Tensor, unmasked: torch.Tensor, min_background_frames: int = 5) -> torch.Tensor:
    """For every frequency the median magnitude over the frames outside the mask (the 20th percentile of all the frames if fewer than
    ``min_background_frames`` are left)."""
    background = torch.where(unmasked, magnitude, torch.full_like(magnitude, float("nan"))).nanmedian(dim=1).values
    fallback = torch.quantile(magnitude, 0.2, dim=1)
    return torch.where(unmasked.sum(dim=1) >= min_background_frames, background, fallback)


def nmf_magnitude(magnitude: torch.Tensor, observed: torch.Tensor, rank: int = 8, iterations: int = 60, seed: int = 0) -> torch.Tensor:
    """A low-rank model ``W H`` of the magnitude fitted only on the observed cells (1 in ``observed``): it predicts the background in
    the masked cells from the spectral shapes that the rest of the clip has (multiplicative updates of the weighted least squares).
    The columns of ``H`` of the frames that are (almost) all masked are copied from random observed frames."""
    generator = torch.Generator().manual_seed(seed)
    v, m = magnitude.double() + 1e-6, observed.double()
    scale = float((v.mean() / rank).sqrt())
    w = torch.rand(v.shape[0], rank, generator=generator, dtype=torch.double) * scale + 1e-3
    h = torch.rand(rank, v.shape[1], generator=generator, dtype=torch.double) * scale + 1e-3
    for _ in range(iterations):
        h = h * (w.T @ (m * v)) / (w.T @ (m * (w @ h)) + 1e-9)
        w = w * ((m * v) @ h.T) / ((m * (w @ h)) @ h.T + 1e-9)
    good = (m.mean(dim=0) >= 0.2).nonzero().flatten()
    bad = (m.mean(dim=0) < 0.2).nonzero().flatten()
    if len(good) == 0:
        return v.float()
    if len(bad):
        h[:, bad] = h[:, good[torch.randint(len(good), (len(bad),), generator=generator)]]
    return (w @ h).float()


def remove_from_stft(waveform: torch.Tensor, mask: torch.Tensor, min_background_frames: int = 5, mode: str = "replace") -> torch.Tensor:
    """The waveform with the masked cells of its STFT (``mask``: ``(n_fft/2 + 1, frames)`` in [0, 1]) filled with a background estimate.

    ``mode``: ``"cap"``: new magnitude ``(1 - m) |X| + m min(|X|, background)`` (it can only take energy away, but leaves the anomaly where
    its cell is already quieter than the background median); ``"replace"``: ``(1 - m) |X| + m background`` (the masked cells get the
    background level, whatever they had); ``"sample"``: every masked cell takes the complex value (magnitude and phase) of the same
    frequency in a random frame outside the mask, so the filled part is real background and not a flat smooth surface; ``"nmf"``: the
    magnitude of the masked cells is predicted by a low-rank model fitted on the cells outside the mask (the phase is the one of
    ``"sample"``).
    """
    if mode not in FILLS:
        raise ValueError(f"mode must be one of {FILLS}, got {mode!r}")
    spectrum, window = _stft(waveform)
    magnitude, phase = spectrum.abs(), torch.angle(spectrum)
    mask = mask.clamp(0, 1)
    unmasked = mask < 0.5
    if mode in ("sample", "nmf"):
        generator = torch.Generator().manual_seed(0)
        quiet = magnitude.argsort(dim=1)[:, : max(min_background_frames, magnitude.shape[1] // 5)]  # the quietest frames, if the mask leaves few
        weights = torch.where(unmasked, 1.0, 0.0)
        few = unmasked.sum(dim=1) < min_background_frames
        weights[few] = 0.0
        weights[few[:, None].expand_as(weights) & torch.zeros_like(weights, dtype=torch.bool).scatter(1, quiet, True)] = 1.0
        source = torch.multinomial(weights, magnitude.shape[1], replacement=True, generator=generator)  # (freq, frames)
        sampled = torch.gather(spectrum, 1, source)
        mixed = spectrum * (1 - mask) + sampled * mask
        if mode == "nmf":
            model = nmf_magnitude(magnitude, unmasked.float())
            mixed = torch.polar(magnitude * (1 - mask) + model * mask, torch.angle(mixed))
        return torch.istft(mixed, N_FFT, hop_length=HOP, win_length=WIN_LENGTH, window=window, center=True, length=waveform.shape[0])
    background = background_magnitude(magnitude, unmasked, min_background_frames)
    fill = torch.minimum(magnitude, background[:, None]) if mode == "cap" else background[:, None].expand_as(magnitude)
    cleaned = magnitude * (1 - mask) + mask * fill
    return torch.istft(torch.polar(cleaned, phase), N_FFT, hop_length=HOP, win_length=WIN_LENGTH, window=window, center=True,
                       length=waveform.shape[0])


def remove_anomaly(waveform: torch.Tensor, mask_fbank: torch.Tensor, min_background_frames: int = 5, mode: str = "replace") -> torch.Tensor:
    """:func:`remove_from_stft` with a mask on the grid of the filterbank (``(frames, 128)`` in [0, 1])."""
    if mode not in FILLS:
        raise ValueError(f"mode must be one of {FILLS}, got {mode!r}")
    return remove_from_stft(waveform, to_stft_grid(mask_fbank, stft_frames(waveform.shape[0])), min_background_frames, mode)


def refine_mask(waveform: torch.Tensor, coarse: torch.Tensor, margin_db: float = 3.0, min_background_frames: int = 5) -> torch.Tensor:
    """The coarse STFT mask restricted to the cells that really stick out: those above the background of their frequency (the median
    outside the mask) by ``margin_db``. The map of the model says where (160 ms x 16 bands), the energy of the clip says what."""
    magnitude = _stft(waveform)[0].abs()
    background = background_magnitude(magnitude, coarse < 0.5, min_background_frames)
    active = (magnitude > background[:, None] * 10 ** (margin_db / 20)).float()[None, None]
    active = F.avg_pool2d(F.max_pool2d(active, 3, stride=1, padding=1), 3, stride=1, padding=1)[0, 0]
    return coarse * active.clamp(0, 1)


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """Scale-invariant SDR in dB of ``estimate`` against ``reference``."""
    estimate, reference = estimate.double() - estimate.double().mean(), reference.double() - reference.double().mean()
    target = (estimate @ reference) / (reference @ reference).clamp_min(1e-20) * reference
    noise = estimate - target
    return float(10 * torch.log10((target @ target).clamp_min(1e-20) / (noise @ noise).clamp_min(1e-20)))


def removal_metrics(cleaned: torch.Tensor, mixture: torch.Tensor, anomaly: torch.Tensor) -> dict[str, float]:
    """How much of the anomaly is gone and how much the background changed.

    The reference is ``mixture - anomaly``, the background that is really inside the clip (the ``bg_clip`` that EnvMix stores is not:
    it is louder by a constant factor). With ``r = cleaned - reference`` and ``alpha = <r, anomaly> / <anomaly, anomaly>``:
    ``anomaly_left`` is alpha (1 = untouched, 0 = gone), ``anomaly_removed_db`` is ``-20 log10 |alpha|`` and ``background_error_db`` is the
    energy of what is left of ``r`` once the anomaly is taken out, relative to the energy of the background (lower = less damage).
    """
    reference = mixture.double() - anomaly.double()
    a = anomaly.double()
    residual = cleaned.double() - reference
    alpha = float((residual @ a) / (a @ a).clamp_min(1e-20))
    rest = residual - alpha * a
    return {"si_sdr_cleaned": si_sdr(cleaned, reference), "anomaly_left": alpha,
            "anomaly_removed_db": float(-20 * np.log10(max(abs(alpha), 1e-3))),
            "background_error_db": float(10 * np.log10(max(float(rest @ rest), 1e-20) / float(reference @ reference)))}


def area_threshold(anomaly_map: np.ndarray, fraction: float) -> float:
    """The threshold that keeps the ``fraction`` of the pixels of the map with the highest values (a mask as big as the true one)."""
    return float(np.quantile(anomaly_map, 1.0 - min(max(fraction, 1e-4), 0.999)))


def adaptive_threshold(anomaly_map: np.ndarray, relative: float) -> float:
    """A threshold that follows the clip: ``relative`` of the way from the lowest to the highest value of its map."""
    low, high = float(np.min(anomaly_map)), float(np.max(anomaly_map))
    return low + relative * (high - low)


def normalise_map(anomaly_map: np.ndarray, tau: float) -> np.ndarray:
    """The map in units of its own threshold (1 = at the threshold), so that the maps of two models can be combined."""
    return anomaly_map / tau if tau > 0 else anomaly_map - tau + 1.0


def fuse_maps(maps: dict[str, np.ndarray], taus: dict[str, float], how: str = "and") -> np.ndarray:
    """The maps of several models in units of their thresholds, combined: ``and`` = the minimum (a pixel counts only if every model
    flags it), ``avg`` = the mean. The fused map is at its threshold where it is 1."""
    stack = np.stack([normalise_map(m, taus[name]) for name, m in maps.items()])
    return stack.min(axis=0) if how == "and" else stack.mean(axis=0)


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
    """A fitted model seen through its anomaly map and score, with what is needed to read them (thresholds, normal scores)."""

    def __init__(self, name: str, model, device: torch.device, thresholds: dict[float, float] | None = None, normal_scores=None):
        self.name, self.model, self.device = name, model, device
        self.thresholds = dict(thresholds or {})  # quantile of the pixels of the normal clips -> threshold of the map
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


def iterative_remove(detector: Detector, waveform: torch.Tensor, tau: float, mode: str, iterations: int, limit: float):
    """Remove, look again and remove what the model still flags: up to ``iterations`` times, and not at all if the score of the clip
    is already at most ``limit`` (a normal clip). Returns the cleaned clip and the union of the masks."""
    current = waveform
    union = torch.zeros(N_FFT // 2 + 1, stft_frames(waveform.shape[0]))
    for _ in range(iterations):
        anomaly_map, score = detector.map_and_score(current)
        if score <= limit:
            break
        mask = to_stft_grid(maps_to_mask(torch.as_tensor(anomaly_map), tau), union.shape[1])
        union = torch.maximum(union, mask)
        current = remove_from_stft(current, mask, mode=mode)
    return current, union


def _fixed_apply(mask: torch.Tensor):
    return lambda w, mode: (remove_from_stft(w, mask, mode=mode), mask)


def _fine_apply(coarse: torch.Tensor, margin_db: float):
    def apply(w, mode):
        mask = refine_mask(w, coarse, margin_db)
        return remove_from_stft(w, mask, mode=mode), mask
    return apply


def _iterative_apply(detector: Detector, tau: float, iterations: int, limit: float):
    return lambda w, mode: iterative_remove(detector, w, tau, mode, iterations, limit)


def _subtract_apply(separator, prompt: str):
    return lambda w, mode: (w - separator(w, prompt), None)


class AudioSepSeparator:
    """AudioSep (text-queried source separation) run in its own environment, one process per clip: the separated source is
    subtracted from the clip. Needs the AudioSep repository, its checkpoint and a python with its requirements."""

    def __init__(self, python: str, repo: Path, checkpoint: Path | None = None, device: str = "cuda"):
        self.python, self.repo, self.device = python, Path(repo).expanduser().resolve(), device
        self.checkpoint = Path(checkpoint).expanduser().resolve() if checkpoint else self.repo / "checkpoint" / "audiosep_base_4M_steps.ckpt"

    def __call__(self, waveform: torch.Tensor, prompt: str) -> torch.Tensor:
        import subprocess
        import tempfile

        import torchaudio.functional as AF
        from scipy.io import wavfile

        scale = 0.9 / float(waveform.abs().max().clamp_min(1e-8))
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "mixture.wav", Path(tmp) / "separated.wav"
            save_wav(source, waveform, scale)
            code = ("import os, sys, torch; repo, ckpt, src, text, out, dev = sys.argv[1:7]; os.chdir(repo); sys.path.insert(0, repo); "
                    "from pipeline import build_audiosep, inference; device = torch.device(dev); "
                    "model = build_audiosep(config_yaml='config/audiosep_base.yaml', checkpoint_path=ckpt, device=device); "
                    "inference(model, src, text, out, device)")
            subprocess.run([self.python, "-c", code, str(self.repo), str(self.checkpoint), str(source), prompt, str(target), self.device], check=True)
            rate, data = wavfile.read(target)
        data = torch.from_numpy(data.astype(np.float32) / (32768.0 if data.dtype == np.int16 else 1.0))
        if data.ndim > 1:
            data = data.mean(dim=1)
        estimate = AF.resample(data, rate, SAMPLE_RATE) / scale
        out = torch.zeros_like(waveform)
        n = min(len(out), len(estimate))
        out[:n] = estimate[:n]
        return out


def process_example(detectors: dict[str, Detector], mixture: torch.Tensor, anomaly: torch.Tensor, truth_mask: np.ndarray, label: str,
                    quantiles: tuple[float, ...] = (0.95,), modes: tuple[str, ...] = ("replace",), normal_quantile: float = 0.95,
                    save=None, main_threshold: str | None = None, main_mode: str | None = None, extras: tuple[str, ...] = EXTRAS,
                    separator=None, prompt: str = "", iterations: int = 3, relative: float = 0.5, margin_db: float = 3.0) -> list[dict]:
    """One row per method, mask and way of filling it.

    Masks: every detector with a threshold per quantile (``q0.95``...) and with the area of the true mask (``area``, which separates the
    quality of the localisation from the choice of the threshold), and ``oracle`` (the true mask: the best a mask can do). ``extras``:
    ``adaptive`` (a threshold that follows the clip: ``rel0.5``), ``fine`` (the mask of the main threshold restricted to the cells that
    stick out of the background in the STFT), ``fusion`` (the maps of the two models combined: ``and``, ``avg``), ``iterative`` (remove,
    look again, remove). ``separator``: AudioSep subtracted from the clip (``audiosep``). Each cleaned clip is judged against the
    background inside the clip (``mixture - anomaly``) and scored again by every detector. As a control the same procedure is applied to
    that background alone: ``control_error_db`` is how much it changes it and ``pct_<model>_control`` where the score of the result falls.
    ``save(key, cleaned)`` is called for the main threshold and filling of each method, and for the oracle.
    """
    reference = mixture - anomaly
    main_threshold = main_threshold or f"q{quantiles[0]:g}"
    main_q = float(main_threshold[1:])
    main_mode = main_mode or modes[0]
    maps, mixture_scores, reference_scores, limits = {}, {}, {}, {}
    for name, detector in detectors.items():
        maps[name], mixture_scores[name] = detector.map_and_score(mixture)
        reference_scores[name] = detector.map_and_score(reference)[1]
        limits[name] = float(np.quantile(detector.normal_scores, normal_quantile)) if len(detector.normal_scores) else float("inf")
    truth = torch.as_tensor(truth_mask).float()
    truth_fraction = float(truth.mean())
    frames = stft_frames(mixture.shape[0])
    on_stft = lambda fbank_mask: to_stft_grid(fbank_mask, frames)  # noqa: E731
    variants: list[dict] = []

    def add(mask, threshold, apply, localisation=None, key=None, fills=None):
        variants.append({"mask": mask, "threshold": threshold, "apply": apply, "loc": localisation or {}, "key": key, "fills": fills})

    for name, detector in detectors.items():
        amap, tensor = maps[name], torch.as_tensor(maps[name])
        for q in quantiles:
            tau = detector.thresholds[q]
            add(name, f"q{q:g}", _fixed_apply(on_stft(maps_to_mask(tensor, tau))), localisation_metrics(amap, truth_mask, tau), name if q == main_q else None)
        tau = area_threshold(amap, truth_fraction)
        add(name, "area", _fixed_apply(on_stft(maps_to_mask(tensor, tau))), localisation_metrics(amap, truth_mask, tau))
        if "adaptive" in extras:
            tau = adaptive_threshold(amap, relative)
            add(name, f"rel{relative:g}", _fixed_apply(on_stft(maps_to_mask(tensor, tau))), localisation_metrics(amap, truth_mask, tau))
        if "fine" in extras:
            coarse = on_stft(maps_to_mask(tensor, detector.thresholds[main_q]))
            add(name, f"q{main_q:g}+fine", _fine_apply(coarse, margin_db), key=f"{name}+fine")
        if "iterative" in extras:
            add(name, f"q{main_q:g}x{iterations}", _iterative_apply(detector, detector.thresholds[main_q], iterations, limits[name]), key=f"{name}+iter")
    if "fusion" in extras and len(detectors) > 1:
        taus = {name: detector.thresholds[main_q] for name, detector in detectors.items()}
        for how in ("and", "avg"):
            fused = fuse_maps(maps, taus, how)
            add("fused", how, _fixed_apply(on_stft(maps_to_mask(torch.as_tensor(fused), 1.0))), localisation_metrics(fused, truth_mask, 1.0),
                "fused" if how == "and" else None)
    add("oracle", "truth", _fixed_apply(on_stft(maps_to_mask(truth, 0.5))), key="oracle")
    if separator is not None:
        add("audiosep", "text", _subtract_apply(separator, prompt), key="audiosep", fills=("subtract",))
    mixture_sdr = si_sdr(mixture, reference)
    reference_energy = float(reference.double() @ reference.double())
    rows = []
    for variant in variants:
        for mode in variant["fills"] or modes:
            cleaned, mask = variant["apply"](mixture, mode)
            control = variant["apply"](reference, mode)[0]
            if save is not None and variant["key"] and mode in (main_mode, "subtract") and variant["threshold"] in (main_threshold, "truth", "and", "text", f"q{main_q:g}+fine", f"q{main_q:g}x{iterations}"):
                save(variant["key"], cleaned)
            row = {"example": label, "mask": variant["mask"], "threshold": variant["threshold"], "mode": mode,
                   "mask_fraction": float(mask.mean()) if mask is not None else float("nan"),
                   "si_sdr_mixture": mixture_sdr, **removal_metrics(cleaned, mixture, anomaly)}
            row["si_sdr_gain"] = row["si_sdr_cleaned"] - mixture_sdr
            diff = control.double() - reference.double()
            row["control_error_db"] = float(10 * np.log10(max(float(diff @ diff), 1e-20) / max(reference_energy, 1e-20)))
            row.update({f"loc_{k}": v for k, v in variant["loc"].items()})
            for name, detector in detectors.items():
                after = detector.map_and_score(cleaned)[1]
                after_control = detector.map_and_score(control)[1]
                row.update({f"score_{name}_mixture": mixture_scores[name], f"score_{name}_cleaned": after,
                            f"score_{name}_background": reference_scores[name],
                            f"pct_{name}_mixture": percentile_of(mixture_scores[name], detector.normal_scores),
                            f"pct_{name}_cleaned": percentile_of(after, detector.normal_scores),
                            f"pct_{name}_control": percentile_of(after_control, detector.normal_scores),
                            f"normal_{name}_cleaned": bool(after <= limits[name])})
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
    panels += [(f"cleaned ({name})", spec(w)) for name, w in cleaned.items()]
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
    """A 16-bit mono wav, with the standard library: torchaudio.save needs a backend that not every installation has."""
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    samples = ((waveform.float() * scale).clamp(-1, 1) * 32767).round().short().numpy()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())


# ---------------------------------------------------------------------------------------------- saved models
def trained_state(name: str, model) -> dict[str, torch.Tensor]:
    """What training changed, without BEATs (a pretrained, frozen copy of it is loaded again with the model).

    PatchCore: its buffers (the memory, the standardisation of the features and of the scores). Dinomaly: the bottleneck and the decoder.
    """
    if name == "dinomaly":
        return {k: v.detach().cpu() for k, v in model.trainable.state_dict().items()}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("feature_extractor.")}


def checkpoint_meta(name: str, cfg: dict, background: str, snr: float, seed: int, train_size: int) -> dict:
    keys = ("backbone", "layers", "patchcore_band_banks", "patchcore_score_std", "patchcore_memory_fraction", "memory_bank_size", "epochs", "dinomaly_distance")
    return {"model": name, "background": background, "snr": float(snr), "seed": int(seed), "train_size": int(train_size),
            "config": {k: cfg.get(k) for k in keys}}


def save_checkpoint(name: str, model, folder: Path, meta: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(trained_state(name, model), folder / f"{name}.pt")
    (folder / f"{name}.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def load_checkpoint(name: str, model, folder: Path, meta: dict) -> bool:
    """Load a saved model into ``model`` if it was trained with the same settings on the same data; False (nothing loaded) otherwise."""
    weights, description = folder / f"{name}.pt", folder / f"{name}.json"
    if not (weights.exists() and description.exists()):
        return False
    if json.loads(description.read_text(encoding="utf-8")) != json.loads(json.dumps(meta)):
        print(f"{name}: the saved model in {folder} was trained with other settings: it is trained again", flush=True)
        return False
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if name == "dinomaly":
        model.trainable.load_state_dict(state)
    else:
        model.load_state_dict(state, strict=False)  # the BEATs weights are not in the file
    return True


def load_base_config(path: Path) -> dict:
    """The config with the defaults of the benchmark (``memory_bank_size``, ``batch_size``...): the runners load it this way, and the
    models are built here without passing through them."""
    from benchmark_common import load_config

    return load_config(Path(path).expanduser())


# ---------------------------------------------------------------------------------------------- the experiment
def summary_table(rows: list[dict], detectors: list[str], mode: str | None = None) -> str:
    """The rows of one way of filling (default: the first found), one per example, mask and threshold."""
    mode = mode or rows[0]["mode"]
    header = ["example", "mask", "threshold", "mask area", "anomaly removed (dB)", "background error (dB)", "control error (dB)"]
    for name in detectors:
        header += [f"{name}: score pct before", f"{name}: score pct after", f"{name}: normal?"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        if r["mode"] not in (mode, "subtract"):
            continue
        cells = [r["example"], r["mask"], r["threshold"], f"{100 * r['mask_fraction']:.0f}%", f"{r['anomaly_removed_db']:.1f}", f"{r['background_error_db']:.1f}",
                 f"{r['control_error_db']:.1f}"]
        for name in detectors:
            cells += [f"{r[f'pct_{name}_mixture']:.2f}", f"{r[f'pct_{name}_cleaned']:.2f}", "yes" if r[f"normal_{name}_cleaned"] else "no"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def aggregate_table(rows: list[dict], detectors: list[str]) -> str:
    """The mean over the examples of every mask, threshold and way of filling."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["mask"], r["threshold"], r["mode"]), []).append(r)
    header = ["mask", "threshold", "fill", "examples", "mask area", "anomaly removed (dB)", "background error (dB)", "control error (dB)", "pixel AUC", "IoU"]
    for name in detectors:
        header += [f"{name}: normal after"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    mean = lambda values: float(np.nanmean([float(v) for v in values if v != ""])) if any(v != "" for v in values) else float("nan")  # noqa: E731
    for (mask, threshold, mode), group in groups.items():
        cells = [mask, threshold, mode, str(len(group)), f"{100 * mean([r['mask_fraction'] for r in group]):.0f}%",
                 f"{mean([r['anomaly_removed_db'] for r in group]):.1f}", f"{mean([r['background_error_db'] for r in group]):.1f}",
                 f"{mean([r['control_error_db'] for r in group]):.1f}", f"{mean([r.get('loc_pixel_auc', '') for r in group]):.2f}", f"{mean([r.get('loc_iou', '') for r in group]):.2f}"]
        for name in detectors:
            cells.append(f"{sum(bool(r[f'normal_{name}_cleaned']) for r in group)}/{len(group)}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_combo(base: dict, background_name: str, snr: float, seed: int, examples: int, output: Path, quantiles: tuple[float, ...],
              models_dir: Path | None = None, retrain: bool = False, modes: tuple[str, ...] = ("sample", "replace", "cap", "nmf"),
              main_quantile: float | None = None, main_mode: str | None = None, extras: tuple[str, ...] = EXTRAS,
              separator=None, iterations: int = 3) -> list[dict]:
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
    folder_models = None if models_dir is None else models_dir / f"{background_name}_snr{snr:g}_seed{seed}"
    for name, cfg in configs.items():
        model = make_model(name, cfg, device, input_size, class_name=background_name)
        meta = checkpoint_meta(name, cfg, background_name, snr, seed, len(train_dataset))
        if folder_models is not None and not retrain and load_checkpoint(name, model, folder_models, meta):
            print(f"{name}: model loaded from {folder_models}", flush=True)
        else:
            fit_model(name, model, train_loader, test_loader, cfg, device, False)
            if folder_models is not None:
                save_checkpoint(name, model, folder_models, meta)
                print(f"{name}: model saved in {folder_models}", flush=True)
        model.eval()
        detectors[name] = Detector(name, model.to(device), device)
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
        pixels = np.concatenate([m.ravel() for m in normal_maps[name]])
        detector.thresholds = {q: float(np.quantile(pixels, q)) for q in quantiles}
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

        background = sample.audio_clip - sample.anomaly_clip  # the background that is really inside the clip (bg_clip is louder: see README)
        rows += process_example(detectors, sample.audio_clip, sample.anomaly_clip, truth, label, quantiles=quantiles, modes=modes, save=save,
                                main_threshold=None if main_quantile is None else f"q{main_quantile:g}", main_mode=main_mode,
                                extras=extras, separator=separator, prompt=str(category).replace("_", " "), iterations=iterations)
        save_wav(folder / "1_before.wav", sample.audio_clip, scale)  # the same scale for every file of the example: the levels are comparable
        save_wav(folder / "5_true_background.wav", background, scale)
        save_wav(folder / "6_anomaly_only.wav", sample.anomaly_clip, scale)
        maps = {name: d.map_and_score(sample.audio_clip)[0] for name, d in detectors.items()}
        try:  # the audio and the numbers are already written: a problem with the figure must not stop the run
            save_figure(folder / "figure.png", sample.audio_clip, background, maps, truth, cleaned_audio)
        except Exception as error:  # noqa: BLE001
            print(f"figure not written for {label}: {error!r}", flush=True)
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
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.9, 0.95, 0.99],
                        help="the map thresholds: these quantiles of the pixels of the normal test clips (every one is evaluated)")
    parser.add_argument("--modes", nargs="+", choices=FILLS, default=["sample", "replace", "cap", "nmf"], help="how the masked part is filled")
    parser.add_argument("--main-quantile", type=float, default=0.95, help="the threshold of the wav files and of the main table")
    parser.add_argument("--main-mode", choices=FILLS, default="sample", help="the way of filling of the wav files and of the main table")
    parser.add_argument("--extras", nargs="*", choices=EXTRAS, default=list(EXTRAS), help="the families of methods to try besides the plain thresholds")
    parser.add_argument("--iterations", type=int, default=3, help="the maximum number of rounds of the iterative removal")
    parser.add_argument("--audiosep-python", help="python of an environment with AudioSep installed: enables the AudioSep separation")
    parser.add_argument("--audiosep-dir", type=Path, help="the AudioSep repository (it holds config/ and checkpoint/)")
    parser.add_argument("--audiosep-checkpoint", type=Path, help="default: <audiosep-dir>/checkpoint/audiosep_base_4M_steps.ckpt")
    parser.add_argument("--models-dir", type=Path, default=Path("results/paper_benchmark/anomaly_removal_models"),
                        help="where the trained models are saved and loaded from (a repeated run does not train again)")
    parser.add_argument("--retrain", action="store_true", help="train again even if a saved model exists")
    parser.add_argument("--no-save", action="store_true", help="do not save or load models")
    args = parser.parse_args()
    base = load_base_config(args.base_config)
    quantiles = tuple(sorted(set(args.quantiles) | {args.main_quantile}))
    separator = None
    if args.audiosep_python:
        if not args.audiosep_dir:
            parser.error("--audiosep-python needs --audiosep-dir")
        separator = AudioSepSeparator(args.audiosep_python, args.audiosep_dir, args.audiosep_checkpoint)
    all_rows: list[dict] = []
    for combo in args.combos:
        background, snr = combo.rsplit(":", 1)
        print(f"\n===== {background} at {float(snr):g} dB", flush=True)
        all_rows += run_combo(base, background, float(snr), args.seed, args.examples_per_combo, args.output_dir, quantiles,
                              None if args.no_save else args.models_dir, args.retrain, tuple(args.modes), args.main_quantile, args.main_mode,
                              tuple(args.extras), separator, args.iterations)
        if all_rows:  # written after every combo: a crash keeps what is done
            args.output_dir.mkdir(parents=True, exist_ok=True)
            fields = list(dict.fromkeys(k for row in all_rows for k in row))
            with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(all_rows)
            detector_names = ["patchcore", "dinomaly"]
            (args.output_dir / "summary.md").write_text(summary_table(all_rows, detector_names, args.main_mode) + "\n", encoding="utf-8")
            (args.output_dir / "aggregate.md").write_text(aggregate_table(all_rows, detector_names) + "\n", encoding="utf-8")
            (args.output_dir / "README.md").write_text(README, encoding="utf-8")
    if all_rows:
        archive = shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir)  # the folder to download
        print(f"\nResults in {args.output_dir}\nZip to download: {archive}")
    else:
        print("\nNo example was produced")


if __name__ == "__main__":
    main()
