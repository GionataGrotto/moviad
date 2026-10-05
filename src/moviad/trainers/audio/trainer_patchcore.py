from typing import Union

import torch
from torch.utils.data import DataLoader

from moviad.models.audio.patchcore.patchcore import PatchCore


class TrainerPatchCore:
    """Fits the memory bank of an audio :class:`PatchCore`.

    Args:
        patchore_model: model to be fitted
        train_dataloader: train dataloader
        test_dataloder: test dataloader (kept for API compatibility, unused)
        device: device to be used for the fit
        force_cpu: run the coreset selection on CPU (e.g. when the GPU is full)
    """

    def __init__(
        self,
        patchore_model: PatchCore,
        train_dataloader: DataLoader,
        test_dataloder: DataLoader,
        device: Union[str, torch.device],
        force_cpu: bool = False,
    ):
        self.patchore_model = patchore_model
        self.train_dataloader = train_dataloader
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.force_cpu = force_cpu

    def train(self, streaming: bool = False):
        """Fit the model. ``streaming`` is ignored: the fit is always memory-bounded."""
        self.patchore_model.fit(self.train_dataloader, force_cpu=self.force_cpu)
