from __future__ import annotations

from itertools import islice

import torch
from tqdm import tqdm

from moviad.models.audio.dinomaly import AudioDinomaly
from moviad.models.dinomaly.dinomaly import DinomalyTrainArgs


class TrainerDinomaly:
    """Minimal trainer for AudioDinomaly and the audio benchmark loaders."""

    def __init__(
        self,
        model: AudioDinomaly,
        train_dataloader,
        device,
        debug: bool = False,
        max_batches: int = 2,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.device = torch.device(device)
        self.debug = debug
        self.max_batches = max(1, int(max_batches))

    def train(self, epochs: int, batch_size: int) -> list[float]:
        self.model.to(self.device)
        steps_per_epoch = len(self.train_dataloader)
        if self.debug:
            steps_per_epoch = min(steps_per_epoch, self.max_batches)
        training_args = DinomalyTrainArgs(batch_size=batch_size, epochs=epochs)
        training_args.total_iters = max(1, epochs * steps_per_epoch)
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
                desc=f"Dinomaly train epoch {epoch + 1}/{epochs}",
            ):
                losses.append(self.model.train_step(batch, training_args))
            epoch_losses.append(sum(losses) / max(1, len(losses)))
        return epoch_losses


__all__ = ["TrainerDinomaly"]
