import os

import torch

from moviad.models.audio.padim.padim import Padim


class PadimTrainer:

    def __init__(self, model: Padim, device, save_path, data_path, class_name):
        """
        Args:
            device: one of the following strings: 'cpu', 'cuda', 'cuda:0', ...
        """
        self.model = model
        self.save_path = save_path
        self.class_name = class_name
        self.device = device

        model.to(device)

    def train(self, train_dataloader, streaming: bool = True):
        """Fit the per-patch Gaussians. ``streaming`` is ignored: the fit is always one streaming pass."""
        print(f"Train Padim. Backbone: {self.model.backbone_model_name}")
        self.model.fit(train_dataloader)

        if self.save_path is not None:
            model_savepath = self.model.get_model_savepath(self.save_path)
            os.makedirs(os.path.dirname(model_savepath), exist_ok=True)
            torch.save(self.model.state_dict(), model_savepath)
