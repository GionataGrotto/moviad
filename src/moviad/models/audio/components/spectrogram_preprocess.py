"""Optional preprocessing of the log-mel spectrograms fed to the pretrained audio backbones.

Two problems of the raw log-mel input are addressed, both off unless requested:

* **Band standardisation.** The stationary background of a machine gives every mel band its own
  mean and spread, so those dominate the features and hide the deviations that matter. Each band
  is z-scored with the statistics of the training set and then mapped back to the overall mean
  and spread, so the dynamic range stays the one the pretrained network was trained on.
* **Dead bands.** DCASE audio is 16 kHz but the backbones expect 44.1/48 kHz, so the mel bands above
  8 kHz contain only the resampling floor (about -100 dB), far outside the training distribution.
  They are replaced by a copy of the last live band.

Spectrograms have shape ``(B, 1, time, mel)``.
"""

from __future__ import annotations

from typing import Iterable

import librosa
import torch
from torch import nn


def first_band_above(cutoff_hz: float, n_mels: int, fmin: float, fmax: float) -> int:
    """Index of the first mel band whose centre frequency is above ``cutoff_hz`` (``n_mels`` if none)."""
    centres = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=fmin, fmax=fmax)[1:-1]
    above = [index for index, centre in enumerate(centres) if centre > cutoff_hz]
    return above[0] if above else n_mels


class SpectrogramPreprocessor(nn.Module):
    def __init__(self, n_mels: int = 64, band_norm: str = "none", dead_band_start: int | None = None):
        super().__init__()
        if band_norm not in ("none", "zscore"):
            raise ValueError(f"band_norm must be 'none' or 'zscore', got {band_norm!r}")
        if dead_band_start is not None and not 1 <= dead_band_start <= n_mels:
            raise ValueError(f"dead_band_start must be in [1, {n_mels}], got {dead_band_start}")
        self.n_mels = n_mels
        self.band_norm = band_norm
        self.dead_band_start = dead_band_start
        self.register_buffer("band_mean", torch.zeros(n_mels))
        self.register_buffer("band_std", torch.ones(n_mels))
        self.register_buffer("overall_mean", torch.zeros(()))
        self.register_buffer("overall_std", torch.ones(()))
        self.register_buffer("fitted", torch.tensor(band_norm == "none"))

    @property
    def active(self) -> bool:
        return self.band_norm != "none" or self.dead_band_start is not None

    def _fill_dead_bands(self, spectrogram: torch.Tensor) -> torch.Tensor:
        start = self.dead_band_start
        if start is None or start >= self.n_mels:
            return spectrogram
        filled = spectrogram.clone()
        filled[..., start:] = spectrogram[..., start - 1 : start]
        return filled

    @torch.no_grad()
    def fit(self, spectrograms: Iterable[torch.Tensor]) -> None:
        """Estimate the per-band statistics of the (dead-band filled) training spectrograms."""
        total = square = 0.0
        count = 0
        for spectrogram in spectrograms:
            x = self._fill_dead_bands(spectrogram).double()
            total = total + x.sum(dim=(0, 1, 2))
            square = square + (x * x).sum(dim=(0, 1, 2))
            count += x.shape[0] * x.shape[1] * x.shape[2]
        if count == 0:
            raise RuntimeError("Cannot fit the spectrogram preprocessor on an empty dataset")
        mean = total / count
        std = (square / count - mean * mean).clamp_min(0).sqrt().clamp_min(1e-6)
        overall_mean = mean.mean()
        overall_second_moment = (square / count).mean()
        self.band_mean.copy_(mean.float())
        self.band_std.copy_(std.float())
        self.overall_mean.copy_(overall_mean.float())
        self.overall_std.copy_((overall_second_moment - overall_mean**2).clamp_min(1e-12).sqrt().float())
        self.fitted.fill_(True)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        spectrogram = self._fill_dead_bands(spectrogram)
        if self.band_norm == "zscore":
            if not bool(self.fitted):
                raise RuntimeError("The spectrogram preprocessor must be fitted before use")
            spectrogram = (spectrogram - self.band_mean) / self.band_std * self.overall_std + self.overall_mean
        return spectrogram
