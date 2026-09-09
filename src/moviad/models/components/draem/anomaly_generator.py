"""Synthetic anomaly generator used to train DRAEM.

DRAEM never sees a real defect during training: anomalies are simulated by
blending a randomly augmented texture into the normal image through a
binarized Perlin noise mask. The simulator is described in Sec. 3.1 of
"DRAEM - A discriminatively trained reconstruction embedding for surface
anomaly detection" (https://arxiv.org/abs/2108.07610).
"""

from __future__ import annotations

import os
import random
from glob import glob

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import functional as TF

from moviad.models.components.simplenet.perlin import generate_perlin_noise

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _adjust_gamma(image: torch.Tensor) -> torch.Tensor:
    return TF.adjust_gamma(image, gamma=random.uniform(0.5, 2.0))


def _adjust_brightness(image: torch.Tensor) -> torch.Tensor:
    image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
    return (image + random.uniform(-0.1, 0.1)).clamp(0.0, 1.0)


def _adjust_sharpness(image: torch.Tensor) -> torch.Tensor:
    return TF.adjust_sharpness(image, random.uniform(0.5, 2.0))


def _adjust_hue_saturation(image: torch.Tensor) -> torch.Tensor:
    image = TF.adjust_hue(image, random.uniform(-0.15, 0.15))
    return TF.adjust_saturation(image, random.uniform(0.5, 1.5))


def _solarize(image: torch.Tensor) -> torch.Tensor:
    return TF.solarize(image, threshold=random.uniform(0.3, 0.9))


def _posterize(image: torch.Tensor) -> torch.Tensor:
    quantized = TF.posterize((image * 255).to(torch.uint8), bits=random.randint(4, 8))
    return quantized.float() / 255.0


def _invert(image: torch.Tensor) -> torch.Tensor:
    return TF.invert(image)


def _autocontrast(image: torch.Tensor) -> torch.Tensor:
    return TF.autocontrast(image)


def _equalize(image: torch.Tensor) -> torch.Tensor:
    equalized = TF.equalize((image * 255).to(torch.uint8))
    return equalized.float() / 255.0


def _rotate(image: torch.Tensor) -> torch.Tensor:
    return TF.rotate(image, angle=random.uniform(-90.0, 90.0))


#: Augmentation pool mirroring the one of the original DRAEM implementation.
AUGMENTATIONS = (
    _adjust_gamma,
    _adjust_brightness,
    _adjust_sharpness,
    _adjust_hue_saturation,
    _solarize,
    _posterize,
    _invert,
    _autocontrast,
    _equalize,
    _rotate,
)


class DraemAnomalyGenerator(nn.Module):
    """Generate simulated anomalies on a batch of anomaly-free images.

    Args:
        anomaly_source_path (str | None): directory holding the out of distribution
            texture images used as anomaly source (the paper uses DTD). When
            ``None`` the textures are synthesised from random Perlin noise, so
            that the model can be trained without any extra dataset.
        perlin_threshold (float): threshold used to binarize the Perlin noise.
        beta_range (tuple[float, float]): range of the opacity factor used to
            blend the texture with the original image.
        anomaly_probability (float): probability of corrupting an image; with
            probability ``1 - anomaly_probability`` the image is left untouched
            and its mask is all zeros.
        num_augmentations (int): how many augmentations of :data:`AUGMENTATIONS`
            are randomly composed and applied to the anomaly source image.
    """

    def __init__(
        self,
        anomaly_source_path: str | None = None,
        perlin_threshold: float = 0.5,
        beta_range: tuple[float, float] = (0.1, 1.0),
        anomaly_probability: float = 0.5,
        num_augmentations: int = 3,
    ) -> None:
        super().__init__()

        self.anomaly_source_path = anomaly_source_path
        self.perlin_threshold = perlin_threshold
        self.beta_range = beta_range
        self.anomaly_probability = anomaly_probability
        self.num_augmentations = num_augmentations
        self.anomaly_source_paths = self._collect_anomaly_sources(anomaly_source_path)

    @staticmethod
    def _collect_anomaly_sources(anomaly_source_path: str | None) -> list[str]:
        if anomaly_source_path is None:
            return []
        if not os.path.isdir(anomaly_source_path):
            raise FileNotFoundError(
                f"Anomaly source directory not found: {anomaly_source_path}"
            )
        paths = sorted(
            path
            for path in glob(os.path.join(anomaly_source_path, "**", "*"), recursive=True)
            if path.lower().endswith(IMG_EXTENSIONS)
        )
        if not paths:
            raise RuntimeError(f"Found 0 anomaly source images in {anomaly_source_path}")
        return paths

    def augment(self, image: torch.Tensor) -> torch.Tensor:
        """Apply ``num_augmentations`` random augmentations to a (C, H, W) image."""
        for augmentation in random.sample(AUGMENTATIONS, k=self.num_augmentations):
            image = augmentation(image)
        return image.clamp(0.0, 1.0)

    def sample_anomaly_source(self, channels: int, height: int, width: int,
                              device: torch.device) -> torch.Tensor:
        """Return an augmented (C, H, W) texture to be used as anomaly source."""
        if self.anomaly_source_paths:
            path = random.choice(self.anomaly_source_paths)
            source = Image.open(path).convert("RGB").resize((width, height))
            source = TF.to_tensor(source).to(device)
            if channels != source.shape[0]:
                source = source.mean(dim=0, keepdim=True).expand(channels, -1, -1)
        else:
            # No texture dataset available: synthesise one out of Perlin noise.
            source = torch.stack(
                [generate_perlin_noise(height, width, device=device) for _ in range(channels)]
            )
            source = (source - source.amin()) / (source.amax() - source.amin() + 1e-8)

        return self.augment(source)

    def generate_perlin_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Return a binary (1, H, W) mask obtained by thresholding Perlin noise."""
        noise = generate_perlin_noise(height, width, device=device).reshape(1, height, width)
        noise = TF.rotate(noise, angle=random.uniform(-90.0, 90.0))
        return (noise > self.perlin_threshold).to(noise.dtype)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Corrupt a batch of anomaly-free images.

        Args:
            images (torch.Tensor): anomaly-free images of shape (B, C, H, W),
                with values in [0, 1].

        Returns:
            tuple: the augmented images (B, C, H, W), the ground truth anomaly
            masks (B, 1, H, W) and the image level labels (B,).
        """
        device = images.device
        _, channels, height, width = images.shape

        augmented_images, masks, labels = [], [], []

        for image in images:
            if random.random() > self.anomaly_probability:
                augmented_images.append(image)
                masks.append(torch.zeros(1, height, width, device=device, dtype=image.dtype))
                labels.append(0.0)
                continue

            mask = self.generate_perlin_mask(height, width, device).to(image.dtype)
            source = self.sample_anomaly_source(channels, height, width, device).to(image.dtype)

            beta = random.uniform(*self.beta_range)
            augmented = image * (1 - mask) + (1 - beta) * source * mask + beta * image * mask

            augmented_images.append(augmented.clamp(0.0, 1.0))
            masks.append(mask)
            labels.append(float(mask.max().item() > 0))

        return (
            torch.stack(augmented_images),
            torch.stack(masks),
            torch.tensor(labels, device=device, dtype=images.dtype),
        )
