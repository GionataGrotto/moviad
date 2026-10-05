"""Anomaly map generator for the audio PatchCore."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from moviad.models.audio.components.feature_ops import gaussian_smooth


class AnomalyMapGenerator(nn.Module):
    """Upsample patch scores to the spectrogram size and smooth them.

    Args:
        sigma: standard deviation of the Gaussian smoothing kernel.
        blur: apply the Gaussian smoothing.
        normalize: min-max normalise the maps over the whole batch. Off by default:
            the statistics of one batch are not those of another, so a normalised
            map has scores that cannot be compared across batches (and any
            per-clip score derived from it, e.g. the DCASE temporal top-k pooling,
            becomes dependent on which clips share the batch).
    """

    def __init__(self, sigma: int = 4, blur: bool = True, normalize: bool = False) -> None:
        super().__init__()
        self.sigma = sigma
        self.blur = blur
        self.normalize = normalize

    def compute_anomaly_map(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        anomaly_map = patch_scores
        if image_size is not None:
            anomaly_map = F.interpolate(
                patch_scores,
                size=(image_size[0], image_size[1]),
                mode="bilinear",
                align_corners=False,
            )
        if self.blur:
            anomaly_map = gaussian_smooth(anomaly_map, self.sigma)
        if self.normalize:
            anomaly_map = AnomalyMapGenerator.rescale(anomaly_map)
        return anomaly_map

    def forward(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        return self.compute_anomaly_map(patch_scores, image_size)

    @staticmethod
    def rescale(x: torch.Tensor) -> torch.Tensor:
        # A batch whose scores are all identical (e.g. a completely uniform
        # patch distance) makes max() - min() == 0, turning every value into
        # NaN and silently corrupting every metric computed downstream.
        span = x.max() - x.min()
        return (x - x.min()) / span if span != 0 else torch.zeros_like(x)
