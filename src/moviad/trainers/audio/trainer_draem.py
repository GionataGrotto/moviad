from __future__ import annotations

from itertools import islice

import torch
from tqdm import tqdm

from moviad.models.audio.draem import AudioDRAEM
from moviad.models.draem.draem import DRAEMTrainArgs


class TrainerDraem:
    """Minimal trainer for AudioDRAEM and the audio benchmark loaders."""

    def __init__(
        self,
        model: AudioDRAEM,
        train_dataloader,
        device,
        debug: bool = False,
        max_batches: int = 2,
        learning_rate: float | None = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.device = torch.device(device)
        self.debug = debug
        self.max_batches = max(1, int(max_batches))
        self.learning_rate = learning_rate

    def train(self, epochs: int, batch_size: int) -> list[float]:
        self.model.to(self.device)

        training_args = DRAEMTrainArgs(batch_size=batch_size, epochs=epochs)
        if self.learning_rate is not None:
            training_args.lr = float(self.learning_rate)
        training_args.init_train(self.model)

        epoch_losses: list[float] = []
        for epoch in range(epochs):
            self.model.train()
            losses = []
            epoch_loader = (
                islice(self.train_dataloader, self.max_batches)
                if self.debug
                else self.train_dataloader
            )
            for batch in tqdm(
                epoch_loader,
                desc=f"DRAEM train epoch {epoch + 1}/{epochs}",
            ):
                losses.append(self.model.train_step(batch, training_args))
            epoch_losses.append(sum(losses) / max(1, len(losses)))
            training_args.lr_scheduler.step()
        return epoch_losses


__all__ = ["TrainerDraem"]
