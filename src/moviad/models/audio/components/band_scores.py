"""Per-frequency-band standardisation of patch anomaly scores.

Patch scores of different frequency bands have different baselines and spreads (a noisy high band
scores higher than a quiet low one even on normal clips), so a max / top-k over the whole map is
dominated by the noisiest bands. Standardising every band with the mean and standard deviation of
the scores of *normal* clips puts the bands on the same scale.

Those statistics must come from normal clips the detector was **not** fitted on: a memory bank
contains its own training patches (their score is about zero) and a Gaussian fitted on the training
set over-fits it, so training scores are biased low. ``is_holdout`` reserves a fraction of the
training batches for this purpose.

Score maps are ``(B, H, W)`` with ``W`` the frequency axis.
"""

from __future__ import annotations

import torch


def holdout_period(fraction: float) -> int:
    """Every ``period``-th batch is held out: ``fraction`` 0.1 -> 10."""
    if not 0 < fraction <= 0.5:
        raise ValueError(f"The held-out fraction must be in (0, 0.5], got {fraction}")
    return max(2, round(1 / fraction))


def is_holdout(batch_index: int, period: int) -> bool:
    return batch_index % period == period - 1


class BandScoreStats:
    """Accumulates the per-band mean and standard deviation of ``(B, H, W)`` score maps."""

    def __init__(self) -> None:
        self.total = None
        self.square_total = None
        self.count = 0

    @torch.no_grad()
    def update(self, scores: torch.Tensor) -> None:
        values = scores.double()
        total = values.sum(dim=(0, 1))
        square = (values * values).sum(dim=(0, 1))
        self.total = total if self.total is None else self.total + total
        self.square_total = square if self.square_total is None else self.square_total + square
        self.count += values.shape[0] * values.shape[1]

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0:
            raise RuntimeError(
                "No held-out normal clips to estimate the per-band score statistics: "
                "use more training batches or a larger held-out fraction"
            )
        mean = self.total / self.count
        std = (self.square_total / self.count - mean * mean).clamp_min(0).sqrt().clamp_min(1e-6)
        return mean.float(), std.float()


def standardize_by_band(scores: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """``(scores - mean_w) / std_w`` for ``(B, H, W)`` scores."""
    if scores.shape[-1] != mean.numel():
        raise ValueError(
            f"The score statistics describe {mean.numel()} frequency bands but the map has {scores.shape[-1]}"
        )
    mean, std = mean.to(scores.device), std.to(scores.device)
    return (scores - mean.view(1, 1, -1)) / std.view(1, 1, -1)
