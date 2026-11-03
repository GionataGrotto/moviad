"""Waveform augmentation of the few target-domain clips, and sampling weights for the two domains.

The development and evaluation sets give about 990 source clips and only 10 target clips per machine. Duplicating a
clip does nothing for a k-center memory bank (a duplicate is never the farthest point) and makes a gradient-trained
model memorise the 10 clips, so the target set is instead enlarged with *new* clips: random gain, equaliser tilt,
circular time shift, small speed / pitch change, noise, and a light mix with a source clip.
"""

from __future__ import annotations

import torch
import torchaudio
from torch.utils.data import Dataset


def _eq_tilt(wave: torch.Tensor, generator: torch.Generator, max_db: float, n_fft: int = 1024) -> torch.Tensor:
    """Multiply the spectrum by a smooth random gain curve (a few control points, linear in dB)."""
    window = torch.hann_window(n_fft)
    spec = torch.stft(wave, n_fft, hop_length=n_fft // 4, window=window, return_complex=True)
    points = (torch.rand(6, generator=generator) * 2 - 1) * max_db
    curve_db = torch.nn.functional.interpolate(points.view(1, 1, -1), size=spec.shape[0], mode="linear", align_corners=True).view(-1)
    spec = spec * (10 ** (curve_db / 20)).view(-1, 1)
    return torch.istft(spec, n_fft, hop_length=n_fft // 4, window=window, length=wave.shape[-1])


def augment_waveform(
    wave: torch.Tensor,
    generator: torch.Generator,
    sample_rate: int = 16000,
    mix: torch.Tensor | None = None,
    probability: float = 0.7,
    gain_db: float = 6.0,
    tilt_db: float = 3.0,
    speed: float = 0.03,
    snr_db: float = 35.0,
    mix_weight: float = 0.2,
) -> torch.Tensor:
    """One random augmentation of a mono waveform ``[T]``; every operation is applied with ``probability``."""
    length = wave.shape[-1]
    chance = lambda: torch.rand(1, generator=generator).item() < probability  # noqa: E731
    uniform = lambda low, high: low + (high - low) * torch.rand(1, generator=generator).item()  # noqa: E731
    wave = wave.clone()
    if chance():
        wave = wave * 10 ** (uniform(-gain_db, gain_db) / 20)
    if chance():
        wave = _eq_tilt(wave, generator, tilt_db)
    if chance():
        factor = uniform(1 - speed, 1 + speed)
        resampled = torchaudio.functional.resample(wave, sample_rate, max(1, int(round(sample_rate * factor))))
        wave = torch.nn.functional.pad(resampled, (0, max(0, length - resampled.shape[-1])))[:length]
    if chance():
        wave = torch.roll(wave, int(torch.randint(0, length, (1,), generator=generator)))
    if chance():
        power = wave.pow(2).mean().clamp_min(1e-12)
        wave = wave + torch.randn(length, generator=generator) * (power / 10 ** (snr_db / 10)).sqrt()
    if mix is not None and chance():
        weight = uniform(0.05, mix_weight)
        mix = torch.roll(mix[:length], int(torch.randint(0, length, (1,), generator=generator)))
        mix = torch.nn.functional.pad(mix, (0, max(0, length - mix.shape[-1])))
        # keep the level of the target clip: the source clip is only a perturbation of its texture
        wave = (1 - weight) * wave + weight * mix * (wave.pow(2).mean().sqrt() / mix.pow(2).mean().sqrt().clamp_min(1e-8))
    return wave


class AugmentedDataset(Dataset):
    """``copies`` random augmentations of every clip of ``base`` (a training set yielding mono waveforms ``[T]``).

    Deterministic: item ``i`` always yields the same clip, so the augmented set is the same in every epoch.
    ``mix_with`` is an optional dataset (typically the source clips) a light mix is drawn from.
    """

    def __init__(self, base: Dataset, copies: int, seed: int = 0, sample_rate: int = 16000, mix_with: Dataset | None = None, **options):
        if copies < 1:
            raise ValueError("copies must be at least 1")
        self.base, self.copies, self.seed, self.sample_rate, self.mix_with, self.options = base, copies, seed, sample_rate, mix_with, options

    def __len__(self) -> int:
        return len(self.base) * self.copies

    def __getitem__(self, index: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        wave = self.base[index % len(self.base)]
        mix = None
        if self.mix_with is not None and len(self.mix_with):
            mix = self.mix_with[int(torch.randint(0, len(self.mix_with), (1,), generator=generator))]
        return augment_waveform(wave, generator, self.sample_rate, mix, **self.options)


def domain_sample_weights(domains: list[str], target_weight: float | str) -> list[float]:
    """Sampling weight per clip: 1 for the source clips and ``target_weight`` for the target ones.

    ``"balance"`` gives the target clips as a group the same total weight as the source clips.
    """
    n_target = sum(d == "target" for d in domains)
    n_source = len(domains) - n_target
    if target_weight == "balance":
        target_weight = n_source / n_target if n_target else 1.0
    return [float(target_weight) if d == "target" else 1.0 for d in domains]
