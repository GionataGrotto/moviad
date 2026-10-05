import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from moviad.models.audio.components.feature_ops import (
    fuse_feature_maps,
    gaussian_smooth,
    temporal_topk_scores,
)


@pytest.mark.parametrize("shape", [(3, 40, 25), (2, 125, 16), (2, 5, 3), (1, 1, 1)])
def test_gaussian_smooth_matches_scipy_including_maps_smaller_than_the_kernel(shape):
    """Small time-frequency maps used to break reflect padding (kernel radius 16)."""
    maps = torch.rand(*shape)
    expected = gaussian_filter(maps.numpy(), sigma=(0, 4, 4))

    smoothed = gaussian_smooth(maps, sigma=4)

    assert smoothed.shape == maps.shape
    np.testing.assert_allclose(smoothed.numpy(), expected, rtol=1e-4, atol=1e-5)


def test_gaussian_smooth_keeps_channel_layout_and_device_dtype():
    maps = torch.rand(2, 1, 30, 20)
    smoothed = gaussian_smooth(maps, sigma=2)
    assert smoothed.shape == maps.shape and smoothed.dtype == maps.dtype
    # a constant map is a fixed point of a normalised smoothing kernel
    constant = torch.full((1, 1, 9, 7), 3.0)
    torch.testing.assert_close(gaussian_smooth(constant, sigma=4), constant)


def test_fuse_feature_maps_covers_every_frame_on_non_integer_ratios():
    fine = torch.ones(2, 3, 69, 16)
    coarse = torch.full((2, 5, 34, 8), 2.0)

    fused = fuse_feature_maps([fine, coarse])

    assert fused.shape == (2, 8, 69, 16)
    assert int((fused[:, 3:].sum(dim=(0, 1, 3)) == 0).sum()) == 0  # no empty trailing frame


def test_fuse_feature_maps_rejects_mismatched_batches():
    with pytest.raises(ValueError, match="Batch size mismatch"):
        fuse_feature_maps([torch.ones(2, 1, 4, 4), torch.ones(3, 1, 2, 2)])


def test_temporal_topk_scores_averages_the_strongest_frequencies():
    maps = torch.tensor([[[[1.0, 5.0, 3.0], [0.0, 0.0, 9.0]]]])  # (B=1, 1, T=2, F=3)
    torch.testing.assert_close(temporal_topk_scores(maps, k=2), torch.tensor([[4.0, 4.5]]))
    # k larger than the number of frequencies is clamped
    torch.testing.assert_close(temporal_topk_scores(maps, k=99), maps.squeeze(1).mean(-1))
