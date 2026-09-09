"""Loss functions of the DRAEM model.

The reconstructive sub-network is trained with an L2 term plus a structural
similarity (SSIM) term, while the discriminative sub-network is trained with a
focal loss on the predicted segmentation logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def _create_window(window_size: int, channels: int, sigma: float = 1.5) -> torch.Tensor:
    window_1d = _gaussian(window_size, sigma).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    """``1 - SSIM`` between two images, averaged over the batch.

    Args:
        window_size (int): side of the gaussian window used to compute the local
            statistics.
        sigma (float): standard deviation of the gaussian window.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", _create_window(window_size, 1, sigma), persistent=False)

    def ssim_map(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the per pixel SSIM map between two (B, C, H, W) images."""
        channels = prediction.shape[1]
        window = self.window.expand(channels, 1, self.window_size, self.window_size)
        window = window.to(device=prediction.device, dtype=prediction.dtype).contiguous()
        padding = self.window_size // 2

        mu1 = F.conv2d(prediction, window, padding=padding, groups=channels)
        mu2 = F.conv2d(target, window, padding=padding, groups=channels)

        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

        sigma1_sq = F.conv2d(prediction * prediction, window, padding=padding, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu2_sq
        sigma12 = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu1_mu2

        c1, c2 = 0.01**2, 0.03**2

        return ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.ssim_map(prediction, target).mean()


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Multi class focal loss on segmentation logits.

    Args:
        logits (torch.Tensor): raw segmentation scores of shape (B, N, H, W).
        target (torch.Tensor): ground truth class indices of shape (B, 1, H, W)
            or (B, H, W).
        alpha (float): weighting factor of the loss.
        gamma (float): focusing parameter down-weighting well classified pixels.

    Returns:
        torch.Tensor: the scalar loss value.
    """
    if target.ndim == logits.ndim:
        target = target.squeeze(1)
    target = target.long()

    log_probabilities = F.log_softmax(logits, dim=1)
    log_probability = log_probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
    probability = log_probability.exp()

    return -(alpha * (1 - probability).pow(gamma) * log_probability).mean()


def draem_loss(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    segmentation_logits: torch.Tensor,
    anomaly_mask: torch.Tensor,
    ssim: SSIMLoss,
) -> torch.Tensor:
    """Overall DRAEM objective: L2 + SSIM on the reconstruction, focal on the mask."""
    l2_loss = F.mse_loss(reconstruction, original)
    ssim_loss = ssim(reconstruction, original)
    segmentation_loss = focal_loss(segmentation_logits, anomaly_mask)
    return l2_loss + ssim_loss + segmentation_loss
