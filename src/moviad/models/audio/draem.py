"""Audio adapter for the visual DRAEM implementation.

The audio benchmark datasets yield waveforms, while DRAEM expects a
three-channel image with values in ``[0, 1]``: both its anomaly simulator
(Perlin blending, solarize, posterize, ...) and its reconstruction target are
defined on that range. This adapter computes the same log-mel spectrogram used
by the other audio models, min-max normalises it per sample, resizes it to the
DRAEM input size, and converts DRAEM's output back to the benchmark audio
contract.

Reference:
    Vitjan Zavrtanik, Matej Kristan, Danijel Skocaj.
    "DRAEM - A discriminatively trained reconstruction embedding for surface
    anomaly detection", ICCV 2021. https://arxiv.org/abs/2108.07610
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from moviad.models.audio.audio_vad_model import AudioVADModel
from moviad.models.draem.draem import DRAEM, DRAEMTrainArgs
from moviad.utilities.audio.audio_feature_extractor import AudioFeatureExtractor

SUPPORTED_SPECTROGRAM_BACKBONES = ("Cnn14", "HTSAT-base")


class AudioDRAEM(DRAEM):
    """DRAEM operating on waveform batches from MIMII/EnvMix/DCASE."""

    def __init__(
        self,
        device: torch.device | str,
        image_size: tuple[int, int] = (256, 256),
        anomaly_source_path: str | None = None,
        spectrogram_backbone: str = "Cnn14",
        base_width_reconstructive: int = DRAEM.DEFAULT_PARAMETERS["base_width_reconstructive"],
        base_width_discriminative: int = DRAEM.DEFAULT_PARAMETERS["base_width_discriminative"],
        anomaly_probability: float = 0.5,
        perlin_threshold: float = 0.5,
    ) -> None:
        image_size = tuple(int(value) for value in image_size)
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError("image_size must contain two positive integers")
        # The discriminative U-Net halves the resolution five times.
        if min(image_size) < 32:
            raise ValueError("image_size must be at least 32x32 for the DRAEM U-Net")

        super().__init__(
            anomaly_source_path=anomaly_source_path,
            input_size=image_size,
            base_width_reconstructive=base_width_reconstructive,
            base_width_discriminative=base_width_discriminative,
            anomaly_probability=anomaly_probability,
            perlin_threshold=perlin_threshold,
        )

        self.device = torch.device(device)
        self.image_size = image_size

        if spectrogram_backbone not in SUPPORTED_SPECTROGRAM_BACKBONES:
            raise ValueError(
                "AudioDRAEM supports the "
                f"{' and '.join(SUPPORTED_SPECTROGRAM_BACKBONES)} spectrogram frontends."
            )

        _, _, self.spectrogram_transform = (
            AudioFeatureExtractor._load_spectrogram_transform(spectrogram_backbone)
        )
        self.spectrogram_transform.eval()
        for parameter in self.spectrogram_transform.parameters():
            parameter.requires_grad = False

    def to(self, device: torch.device | str):
        device = torch.device(device)
        super().to(device)
        self.device = device
        self.spectrogram_transform.to(device)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        # The log-mel frontend is frozen: it must never collect batch statistics.
        self.spectrogram_transform.eval()
        return self

    @staticmethod
    def _batch_input(batch: torch.Tensor | tuple | list) -> torch.Tensor:
        return AudioVADModel.batch_input(batch)

    @staticmethod
    def _normalize(spectrogram: torch.Tensor) -> torch.Tensor:
        """Min-max normalise every sample of a (B, C, H, W) batch to [0, 1].

        Log-mel spectrograms are expressed in dB and span a negative range, but
        DRAEM's anomaly simulator and reconstruction loss assume [0, 1] inputs.
        """
        flattened = spectrogram.flatten(start_dim=1)
        minimum = flattened.amin(dim=1).reshape(-1, 1, 1, 1)
        maximum = flattened.amax(dim=1).reshape(-1, 1, 1, 1)
        return (spectrogram - minimum) / (maximum - minimum).clamp_min(1e-8)

    def _to_image(self, batch: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Convert a waveform or log-mel tensor to a 3-channel DRAEM input."""
        if not isinstance(batch, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor, got {type(batch)!r}")

        batch = batch.to(self.device, dtype=torch.float32)

        # Benchmark training/test datasets provide [B, samples].  Supporting
        # [B, 1, samples] as well makes the adapter safe for custom loaders.
        if batch.ndim == 2:
            spectrogram = self.spectrogram_transform(batch)
        elif batch.ndim == 3 and batch.shape[1] == 1 and batch.shape[-1] > 256:
            spectrogram = self.spectrogram_transform(batch[:, 0, :])
        elif batch.ndim == 4:
            # Useful for callers that already computed a [B, C, H, W] feature
            # map; this path is not used by the benchmark loaders.
            spectrogram = batch
        elif batch.ndim == 3:
            spectrogram = batch.unsqueeze(1)
        else:
            raise ValueError(
                "AudioDRAEM expects [B, samples], [B, 1, samples], "
                f"or [B, C, H, W], got shape {tuple(batch.shape)}"
            )

        if spectrogram.ndim != 4:
            raise RuntimeError(
                "The spectrogram frontend must return a 4D tensor, got "
                f"shape {tuple(spectrogram.shape)}"
            )

        original_size = (int(spectrogram.shape[-2]), int(spectrogram.shape[-1]))

        # The log-mel frontend is mono.  If a custom frontend returns more
        # channels, average them before replicating to RGB.
        if spectrogram.shape[1] != 1:
            spectrogram = spectrogram.mean(dim=1, keepdim=True)

        spectrogram = F.interpolate(
            spectrogram,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return self._normalize(spectrogram).repeat(1, 3, 1, 1), original_size

    def forward(self, batch: torch.Tensor):
        """Run DRAEM on the spectrogram of a waveform batch.

        Returns:
            training: the reconstructed spectrogram and the segmentation logits.
            inference: the anomaly maps at the original spectrogram resolution,
            the clip level scores and the per time frame scores.
        """
        image, original_size = self._to_image(batch)
        output = DRAEM.forward(self, image)

        if self.training:
            # DRAEM's training contract is (reconstruction, segmentation_logits).
            return output

        anomaly_maps, _ = output
        anomaly_maps = F.interpolate(
            anomaly_maps,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        anomaly_scores = anomaly_maps.flatten(start_dim=1).amax(dim=1)
        # Audio spectrograms use (time, frequency) spatial axes. For the
        # temporal metric, pool the five highest-frequency anomaly values and
        # keep one score for every time frame. The (batch, time) shape matches
        # the temporal ground truth masks and the other audio models.
        map_without_channel = anomaly_maps.squeeze(1)
        top_k = min(5, map_without_channel.shape[2])
        temporal_scores = map_without_channel.topk(top_k, dim=2).values.mean(dim=2)
        return anomaly_maps, anomaly_scores, temporal_scores

    def train_step(self, batch, training_args: DRAEMTrainArgs):
        if training_args.optimizer is None:
            training_args.init_train(self)

        waveforms = self._batch_input(batch)
        images, _ = self._to_image(waveforms)

        # Simulate the anomalies on the normalised spectrogram and let DRAEM
        # restore and segment them.
        augmented_images, anomaly_masks, _ = self.anomaly_generator(images)
        reconstruction, segmentation_logits = DRAEM.forward(self, augmented_images)

        loss = training_args.loss_function(
            reconstruction, images, segmentation_logits, anomaly_masks, self.ssim_loss
        )

        training_args.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        training_args.optimizer.step()

        return float(loss.detach().item())


__all__ = ["AudioDRAEM", "DRAEMTrainArgs"]
