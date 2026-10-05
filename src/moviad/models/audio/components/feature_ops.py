"""Tensor helpers shared by the audio PaDiM / PatchCore heads."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def fuse_feature_maps(maps: Sequence[torch.Tensor], mode: str = "nearest") -> torch.Tensor:
    """Resize every ``(B, C_i, H_i, W_i)`` map to the largest grid and concatenate channels.

    Audio feature maps rarely have integer resolution ratios (the time axis follows
    the clip length), so the coarse maps are replicated with nearest-neighbour
    resizing instead of unfold/fold tricks that assume exact multiples.
    """
    maps = list(maps)
    if not maps:
        raise ValueError("fuse_feature_maps needs at least one feature map")
    batch = maps[0].shape[0]
    if any(m.shape[0] != batch for m in maps):
        raise ValueError(
            f"Batch size mismatch between feature maps: {[m.shape[0] for m in maps]}"
        )
    height = max(m.shape[-2] for m in maps)
    width = max(m.shape[-1] for m in maps)
    resized = [
        m if tuple(m.shape[-2:]) == (height, width) else F.interpolate(m, size=(height, width), mode=mode)
        for m in maps
    ]
    return torch.cat(resized, dim=1)


def _gaussian_kernel(sigma: float, truncate: float, device, dtype) -> torch.Tensor:
    radius = int(truncate * sigma + 0.5)
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _symmetric_index(length: int, radius: int, device) -> torch.Tensor:
    """Indices that pad an axis like ``scipy.ndimage`` mode ``reflect`` (d c b a | a b c d).

    Unlike ``F.pad(mode="reflect")`` this never fails when the kernel radius is
    as large as the axis (small time-frequency maps), because the reflection is
    folded with period ``2 * length``.
    """
    positions = torch.arange(-radius, length + radius, device=device)
    folded = positions.remainder(2 * length)
    return torch.where(folded >= length, 2 * length - 1 - folded, folded)


def gaussian_smooth(maps: torch.Tensor, sigma: float, truncate: float = 4.0) -> torch.Tensor:
    """Separable Gaussian blur over the last two dimensions, matching ``scipy.ndimage.gaussian_filter``.

    Accepts ``(H, W)``, ``(B, H, W)`` or ``(B, C, H, W)`` tensors and returns the
    same shape. Stays on the input device (no CPU round trip).
    """
    if sigma <= 0:
        return maps
    original_shape = maps.shape
    x = maps.reshape(-1, 1, *original_shape[-2:])
    if not torch.is_floating_point(x):
        x = x.float()
    kernel = _gaussian_kernel(sigma, truncate, x.device, x.dtype)
    radius = kernel.numel() // 2
    height, width = x.shape[-2:]
    x = x.index_select(2, _symmetric_index(height, radius, x.device))
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    x = x.index_select(3, _symmetric_index(width, radius, x.device))
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    return x.reshape(original_shape)


def temporal_topk_scores(anomaly_maps: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Per time frame score: mean of the ``k`` largest frequency responses.

    ``anomaly_maps`` is ``(B, 1, T, F)`` or ``(B, T, F)``; the result is ``(B, T)``.
    """
    maps = anomaly_maps.squeeze(1) if anomaly_maps.ndim == 4 else anomaly_maps
    k = max(1, min(int(k), maps.shape[-1]))
    return maps.topk(k, dim=-1).values.mean(dim=-1)
