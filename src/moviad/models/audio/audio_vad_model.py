from __future__ import annotations

from typing import Any

import torch

from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel


class AudioVADModel(VADModel):
    """Base class for audio anomaly-detection models.

    Audio models share the same public contract as visual ``VADModel`` classes,
    while accepting waveform batches and optionally owning non-Module feature
    extractors.
    """

    def __init__(
        self,
        feature_extractor: Any | None = None,
        input_size: tuple[int, int] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.input_size = input_size
        self.device = torch.device(device) if device is not None else torch.device("cpu")

    @staticmethod
    def batch_input(batch):
        return batch[0] if isinstance(batch, (tuple, list)) else batch

    def to(self, device: torch.device | str):
        device = torch.device(device)
        super().to(device)
        self.device = device

        if self.feature_extractor is not None and hasattr(self.feature_extractor, "to"):
            self.feature_extractor.to(device)

        return self

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement train_step")

    def train_chunk(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        training_args: TrainingArgs,
    ):
        return self.train_epoch(0, train_dataloader, training_args)

    def reset_model(self):
        pass

    def save(self, save_path: str):
        torch.save(self.state_dict(), save_path)

    def get_model_size(self):
        if hasattr(self, "get_model_size_and_macs"):
            sizes, total_size = self.get_model_size_and_macs()
            return {"components": sizes, "total": total_size}
        return super().get_model_size()
