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
from moviad.utilities.audio.audio_feature_extractor import AudioFeatureExtractor


class AudioDinomaly(Dinomaly):
    """Dinomaly operating on waveform batches from MIMII/EnvMix."""

    def __init__(
        self,
        encoder_name: str,
        device: torch.device | str,
        image_size: tuple[int, int] = (224, 224),
        pretrained: bool = True,
        spectrogram_backbone: str = "Cnn14",
        distance: str = "cosine",
    ) -> None:
        """``distance`` is how encoder and decoder features are compared at test time: ``"cosine"``
        (Dinomaly's own map, ``1 - cos``) or ``"l2"`` (euclidean norm of the difference, which keeps
        the feature magnitude that the cosine discards)."""
        super().__init__(encoder_name=encoder_name, pretrained=pretrained)
        if distance not in ("cosine", "l2"):
            raise ValueError(f"distance must be 'cosine' or 'l2', got {distance!r}")
        self.distance = distance
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
        self.spectro_preprocessor = None  # optional SpectrogramPreprocessor

    def set_spectrogram_preprocessor(self, preprocessor) -> None:
        self.spectro_preprocessor = preprocessor.to(self.device)

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

        if self.spectro_preprocessor is not None and batch.ndim != 4:
            spectrogram = self.spectro_preprocessor(spectrogram)

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
        if not self.training and self.distance == "l2":
            output = (self._l2_map(image), None)
        else:
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
        # (batch, time), matching the temporal ground truth masks and the
        # convention of PatchCore, CFA and STFPM.
        map_without_channel = anomaly_maps.squeeze(1)
        top_k = min(5, map_without_channel.shape[2])
        temporal_scores = map_without_channel.topk(top_k, dim=2).values.mean(dim=2)
        return anomaly_maps, anomaly_scores, temporal_scores

    @torch.no_grad()
    def _l2_map(self, image: torch.Tensor) -> torch.Tensor:
        """Euclidean distance between encoder and decoder features, ``(B, H, W)`` at the image size.

        The visual Dinomaly returns the encoder and decoder features when ``training`` is set and
        builds its cosine map otherwise. Setting the flag on this module alone (not on its blocks,
        which stay in eval mode, so dropout and similar layers remain off) gives access to the
        features without duplicating the encoder-decoder pass.
        """
        self.training = True
        try:
            encoder_features, decoder_features = super().forward(image)
        finally:
            self.training = False
        maps = [
            F.interpolate(
                torch.linalg.vector_norm(encoder - decoder, dim=1).unsqueeze(1),
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            for encoder, decoder in zip(encoder_features, decoder_features)
        ]
        return torch.cat(maps, dim=1).mean(dim=1)

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
