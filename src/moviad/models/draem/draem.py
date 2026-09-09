"""DRAEM: a discriminatively trained reconstruction embedding for surface anomaly detection.

Reference:
    Vitjan Zavrtanik, Matej Kristan, Danijel Skocaj.
    "DRAEM - A discriminatively trained reconstruction embedding for surface
    anomaly detection", ICCV 2021. https://arxiv.org/abs/2108.07610

The model is trained only on anomaly-free images: defects are simulated on the
fly by :class:`DraemAnomalyGenerator`. The reconstructive sub-network learns to
restore the original appearance of the corrupted image, and the discriminative
sub-network learns to segment the corrupted regions from the joint
representation of the corrupted image and of its reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from moviad.models.components.draem.anomaly_generator import DraemAnomalyGenerator
from moviad.models.components.draem.network import (
    DiscriminativeSubNetwork,
    ReconstructiveSubNetwork,
)
from moviad.models.draem.loss_functions import SSIMLoss, draem_loss
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel


@dataclass
class DRAEMTrainArgs(TrainingArgs):
    """Training arguments of DRAEM.

    Args:
        lr (float): learning rate of the Adam optimizer.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler | None): scheduler
            stepped at the end of every epoch. When ``None`` a ``MultiStepLR``
            decaying the learning rate at 80% and 90% of the training is built.
    """

    lr: float = 1e-4
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def init_train(self, model: VADModel):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": model.reconstructive_subnetwork.parameters(), "lr": self.lr},
                    {"params": model.discriminative_subnetwork.parameters(), "lr": self.lr},
                ]
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[int(self.epochs * 0.8), int(self.epochs * 0.9)],
                gamma=0.2,
            )
        if self.loss_function is None:
            self.loss_function = draem_loss

    def __to_dict__(self):
        args = super().__to_dict__()
        args["lr"] = self.lr
        args["lr_scheduler"] = (
            self.lr_scheduler.__class__.__name__ if self.lr_scheduler else None
        )
        return args


class DRAEM(VADModel):
    """DRAEM anomaly detection model.

    Args:
        anomaly_source_path (str | None): directory of the out of distribution
            texture images used to simulate the anomalies (the paper uses the
            DTD dataset). When ``None`` the textures are synthesised from Perlin
            noise, so no additional dataset is required.
        input_size (tuple[int, int]): spatial size of the input images. The
            anomaly maps are produced at this same resolution.
        base_width_reconstructive (int): width of the reconstructive sub-network.
        base_width_discriminative (int): width of the discriminative sub-network.
        anomaly_probability (float): probability of corrupting a training image.
        perlin_threshold (float): threshold binarizing the Perlin noise mask.
        score_smoothing_kernel_size (int): side of the mean filter applied to the
            anomaly map before taking its maximum as image level score.
    """

    DEFAULT_PARAMETERS = {
        "epochs": 700,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "input_size": (256, 256),
        "base_width_reconstructive": 128,
        "base_width_discriminative": 64,
    }

    def __init__(
        self,
        anomaly_source_path: str | None = None,
        input_size: tuple[int, int] = DEFAULT_PARAMETERS["input_size"],
        base_width_reconstructive: int = DEFAULT_PARAMETERS["base_width_reconstructive"],
        base_width_discriminative: int = DEFAULT_PARAMETERS["base_width_discriminative"],
        anomaly_probability: float = 0.5,
        perlin_threshold: float = 0.5,
        score_smoothing_kernel_size: int = 21,
    ):
        super().__init__()

        self.device = torch.device("cpu")
        self.input_size = input_size
        self.score_smoothing_kernel_size = score_smoothing_kernel_size

        self.reconstructive_subnetwork = ReconstructiveSubNetwork(
            in_channels=3, out_channels=3, base_width=base_width_reconstructive
        )
        self.discriminative_subnetwork = DiscriminativeSubNetwork(
            in_channels=6, out_channels=2, base_width=base_width_discriminative
        )

        self.anomaly_generator = DraemAnomalyGenerator(
            anomaly_source_path=anomaly_source_path,
            perlin_threshold=perlin_threshold,
            anomaly_probability=anomaly_probability,
        )
        self.ssim_loss = SSIMLoss()

    def to(self, device: torch.device):
        super().to(device)
        self.device = device
        return self

    def forward(self, batch: torch.Tensor):
        """Run both sub-networks on a batch of images.

        Returns:
            training: the reconstructed images and the segmentation logits.
            inference: the anomaly maps (B, 1, H, W) and the anomaly scores (B,).
        """
        reconstruction = self.reconstructive_subnetwork(batch)
        joined = torch.cat((batch, reconstruction), dim=1)
        segmentation_logits = self.discriminative_subnetwork(joined)

        if self.training:
            return reconstruction, segmentation_logits

        return self.post_process(segmentation_logits)

    def post_process(self, segmentation_logits: torch.Tensor):
        """Turn the segmentation logits into anomaly maps and image level scores."""
        anomaly_maps = torch.softmax(segmentation_logits, dim=1)[:, 1:2, :, :]

        kernel_size = min(
            self.score_smoothing_kernel_size, anomaly_maps.shape[-2], anomaly_maps.shape[-1]
        )
        if kernel_size % 2 == 0:
            kernel_size -= 1

        smoothed = F.avg_pool2d(
            anomaly_maps, kernel_size, stride=1, padding=kernel_size // 2
        )
        anomaly_scores = torch.amax(smoothed.reshape(smoothed.shape[0], -1), dim=1)

        return anomaly_maps, anomaly_scores

    def train_epoch(self, epoch, train_dataloader, training_args: DRAEMTrainArgs):
        avg_batch_loss = 0

        for batch in tqdm(train_dataloader):
            avg_batch_loss += self.train_step(batch, training_args)

        avg_batch_loss /= len(train_dataloader)

        if getattr(training_args, "lr_scheduler", None) is not None:
            training_args.lr_scheduler.step()

        return avg_batch_loss

    def train_step(self, batch: torch.Tensor, training_args: DRAEMTrainArgs):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(self.device)

        # simulate the anomalies and let the model restore and segment them
        augmented_images, anomaly_masks, _ = self.anomaly_generator(images)
        reconstruction, segmentation_logits = self.forward(augmented_images)

        loss = training_args.loss_function(
            reconstruction, images, segmentation_logits, anomaly_masks, self.ssim_loss
        )

        training_args.optimizer.zero_grad()
        loss.backward()
        training_args.optimizer.step()

        return loss.item()

    def save(self, save_path: str):
        torch.save(self.state_dict(), save_path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def get_model_size(self):
        reconstructive_params = sum(p.numel() for p in self.reconstructive_subnetwork.parameters())
        discriminative_params = sum(p.numel() for p in self.discriminative_subnetwork.parameters())
        bytes_per_param = 4

        return {
            "reconstructive_subnetwork": (reconstructive_params * bytes_per_param) / (1024**2),
            "discriminative_subnetwork": (discriminative_params * bytes_per_param) / (1024**2),
            "total": ((reconstructive_params + discriminative_params) * bytes_per_param)
            / (1024**2),
        }
