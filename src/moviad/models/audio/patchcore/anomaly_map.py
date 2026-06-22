"""Anomaly Map Generator for the PatchCore model implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F 
from torchvision.transforms import GaussianBlur

class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.

    Args:
        The anomaly map is upsampled to this dimension.
        sigma (int, optional): Standard deviation for Gaussian Kernel.
            Defaults to ``4``.
    """

    def __init__(
        self,
        sigma: int = 4,
        blur=True,
    ) -> None:
        super().__init__()
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        if blur:
            self.blur = GaussianBlur(kernel_size, sigma)
        else:
            self.blur = None
        #self.blur = GaussianBlur(4)

    def compute_anomaly_map(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        """Pixel Level Anomaly Heatmap.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
            image_size (tuple[int, int] | torch.Size, optional): Size of the input image.
                The anomaly map is upsampled to this dimension.
                Defaults to None.

        Returns:
            Tensor: Map of the pixel-level anomaly scores
        """
        if image_size is None:
            anomaly_map = patch_scores
        else:
            anomaly_map = F.interpolate(patch_scores, size=(image_size[0], image_size[1]), mode="bilinear", align_corners=False)
        if self.blur:
            anomaly_map = self.blur(anomaly_map)
        return AnomalyMapGenerator.rescale(anomaly_map)

    def forward(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        """Return anomaly_map and anomaly_score.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
            image_size (tuple[int, int] | torch.Size, optional): Size of the input image.
                The anomaly map is upsampled to this dimension.
                Defaults to None.

        Example:
            >>> anomaly_map_generator = AnomalyMapGenerator()
            >>> map = anomaly_map_generator(patch_scores=patch_scores)

        Returns:
            Tensor: anomaly_map
        """
        return self.compute_anomaly_map(patch_scores, image_size)

    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())