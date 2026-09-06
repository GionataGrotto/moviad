"""Audio adapter for the visual Dinomaly implementation.

The audio benchmark datasets yield waveforms, while Dinomaly expects a
three-channel image.  This adapter computes the same Cnn14 log-mel
spectrogram used by the other audio models, resizes it to the ViT input size,
and converts Dinomaly's output back to the benchmark audio contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from moviad.models.dinomaly.dinomaly import Dinomaly, DinomalyTrainArgs
from moviad.models.audio.audio_vad_model import AudioVADModel
from moviad.utilities.audio.audio_feature_exctractor import AudioFeatureExtractor


class AudioDinomaly(Dinomaly):
    """Dinomaly operating on waveform batches from MIMII/EnvMix."""

    def __init__(
        self,
        encoder_name: str,
        device: torch.device | str,
        image_size: tuple[int, int] = (224, 224),
        pretrained: bool = True,
        spectrogram_backbone: str = "Cnn14",
    ) -> None:
        super().__init__(encoder_name=encoder_name, pretrained=pretrained)
        self.device = torch.device(device)
        self.image_size = tuple(int(value) for value in image_size)
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain two positive integers")

        if spectrogram_backbone not in {"Cnn14", "HTSAT-base"}:
            raise ValueError(
                "AudioDinomaly supports the Cnn14 and HTSAT-base spectrogram frontends."
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
        self.spectrogram_transform.eval()
        return self

    @staticmethod
    def _batch_input(batch: torch.Tensor | tuple | list) -> torch.Tensor:
        return AudioVADModel.batch_input(batch)

    def _to_image(self, batch: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Convert a waveform or log-mel tensor to a 3-channel ViT input."""
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
                "AudioDinomaly expects [B, samples], [B, 1, samples], "
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
        return spectrogram.repeat(1, 3, 1, 1), original_size

    def forward(self, batch: torch.Tensor):
        image, original_size = self._to_image(batch)
        output = super().forward(image)

        if self.training:
            # Dinomaly's training contract is (encoder_features, decoder_features).
            return output

        anomaly_maps, _ = output
        if anomaly_maps.ndim == 3:
            anomaly_maps = anomaly_maps.unsqueeze(1)
        anomaly_maps = F.interpolate(
            anomaly_maps,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        anomaly_scores = anomaly_maps.flatten(start_dim=1).amax(dim=1)
        # Audio spectrograms use (time, frequency) spatial axes. For the
        # temporal metric, pool the five highest-frequency anomaly values and
        # keep one score for every time frame.
        top_k = min(5, anomaly_maps.shape[3])
        temporal_scores = anomaly_maps.topk(top_k, dim=3).values.mean(dim=3)
        return anomaly_maps, anomaly_scores, temporal_scores

    def train_step(self, batch, training_args: DinomalyTrainArgs):
        if training_args.optimizer is None or training_args.lr_scheduler is None:
            training_args.init_train(self)

        batch = self._batch_input(batch).to(self.device)
        encoder_features, decoder_features = self(batch)

        p_final = 0.9
        p = min(p_final * self.train_steps / 1000, p_final)
        loss = training_args.loss_function(
            encoder_features,
            decoder_features,
            p=p,
            factor=0.1,
        )

        training_args.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainable.parameters(), max_norm=0.1)
        training_args.optimizer.step()
        training_args.lr_scheduler.step()
        self.train_steps += 1
        return float(loss.detach().item())


__all__ = ["AudioDinomaly", "DinomalyTrainArgs"]
