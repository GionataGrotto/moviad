from __future__ import annotations
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import GaussianBlur
from einops import rearrange
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from moviad.models.components.cfa.descriptor import Descriptor
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs
from moviad.models.cfa.cfa_loss import soft_boundary
from moviad.utilities.get_sizes import count_params, params_to_mb

@dataclass
class CFATrainArgs(TrainingArgs):

    def init_train(self, model: VADModel):
        if self.optimizer is None:
            learning_rate = 1e-3
            weight_decay = 5e-4
            self.optimizer = AdamW(params=model.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay,
                            amsgrad=True)
        if self.loss_function is None:
            self.loss_function = soft_boundary


class CFA(VADModel):

    def __init__(
            self,
            feature_extractor: CustomFeatureExtractor,
            backbone: str,
            gamma_c:int = 1,
            gamma_d:int = 1,
            feature_maps_channels: int = 1792,
        ):

        """
        Args:
            feature_extractor () : feature extractor to be used
            training_dataloader (torch.utils.data.DataLoader) : training dataloader
            backbone (str) : name of the used backbone in the feature extractor
            gamma_c (int) : model parameter
            gamma_d (int) : model parameter
            device (torch.device) : device where to run the model
        """

        super().__init__()

        self.memory_bank = 0
        self.nu = 1e-3
        self.scale = None

        self.gamma_c = gamma_c
        self.gamma_d = gamma_d
        self.alpha = 1e-1
        self.K = 3
        self.J = 3
        self.r = nn.Parameter(1e-5*torch.ones(1), requires_grad=True)

        self.device = feature_extractor.device

        self.backbone = backbone
        self.feature_maps_shape: tuple = None

        self.feature_extractor = feature_extractor
        self.feature_maps_channels = feature_maps_channels
        self.Descriptor = Descriptor(self.gamma_d, self.feature_maps_channels, self.backbone, self.device).to(self.device)

    def to(self, device: torch.device):
        super().to(device)
        self.feature_extractor.to(device)
        self.device = device

    def initialize_memory_bank(self, training_dataloader: DataLoader):
        """
        Initialize the memory bank

        Args:
            training_dataloader (DataLoader) : training dataloader
        """

        memory_bank = self.init_centroid(self.feature_extractor, training_dataloader)
        memory_bank = rearrange(memory_bank, 'b c h w -> (b h w) c').detach()

        if self.gamma_c > 1:
            memory_bank = memory_bank.cpu().detach().numpy()
            memory_bank = KMeans(n_clusters=(self.scale**2)//self.gamma_c, max_iter=3000).fit(memory_bank).cluster_centers_
            memory_bank = torch.Tensor(memory_bank).to(self.device)

        memory_bank = memory_bank.transpose(-1, -2).detach()
        return memory_bank

    def forward(self, x: torch.Tensor) -> float | tuple[torch.Tensor, torch.Tensor]:
        """
        CFA forward pass

        Args:
            x (torch.Tensor) : batch of images

        Returns:
            loss (float) if the model is in training mode
            else a tuple:
                [0] : tensor with the anomaly maps (one for every input image)
                [1] : tensor with image level anomaly scores (one for every input image)
        """

        p = self.feature_extractor(x)
        if isinstance(p, dict):
            p = list(p.values())

        if not self.feature_maps_shape:
            self.feature_maps_shape = (1, self.feature_maps_channels, p[0].shape[2], p[0].shape[3])

        phi_p = self.Descriptor(p)
        phi_p = rearrange(phi_p, 'b c h w -> b (h w) c')

        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers  = torch.sum(torch.pow(self.memory_bank, 2), 0, keepdim=True)
        f_c      = 2 * torch.matmul(phi_p, (self.memory_bank))
        dist     = features + centers - f_c
        dist     = torch.sqrt(dist)

        n_neighbors = self.K
        dist     = dist.topk(n_neighbors, largest=False).values

        dist = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
        dist = dist.unsqueeze(-1)

        if self.training:
            return phi_p
        else:
            self.scale = p[0].size(2)
            scores = rearrange(dist, 'b (h w) c -> b c h w', h=self.scale).cpu().detach()

            heatmaps = torch.mean(scores, dim=1)
            heatmaps = CFA.upsample(heatmaps, size=x.size(2), mode="bilinear")
            heatmaps = CFA.gaussian_smooth_torch(heatmaps, sigma=4)
            # NOTE: this used to also compute `CFA.rescale(heatmaps)` into
            # img_scores first, but that result was immediately overwritten
            # below and never used -- dead code, removed. If a normalized
            # (rescaled) image score was actually intended here instead of
            # the raw max distance, flag it for a decision before changing
            # what the model reports.
            img_scores = scores.reshape(scores.shape[0], -1).max(axis=1).values

            if len(heatmaps.shape) == 2:
                return heatmaps.view(1,1,heatmaps.shape[0], heatmaps.shape[1]), img_scores
            else:
                return heatmaps.unsqueeze(dim=1), img_scores

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):
        """
        Training step

        Args:
            x (torch.Tensor) : batch of images

        Returns:
            loss (float) : CFA loss
        """

        self.train()
        loss = training_args.loss_function(self(batch.to(self.device)), self.memory_bank, self.K, self.J, self.r, self.alpha, self.nu)
        training_args.optimizer.zero_grad()
        loss.backward()
        training_args.optimizer.step()

        return loss.item()

    def train_epoch(self, epoch: int, train_dataloader: torch.utils.data.DataLoader, training_args: TrainingArgs):

        self.train()

        if epoch == 0:
            self.memory_bank = nn.Parameter(self.initialize_memory_bank(train_dataloader), requires_grad=False)

        avg_batch_loss = 0

        # train the model
        for batch in tqdm(train_dataloader):
            avg_batch_loss += self.train_step(batch, training_args)

        avg_batch_loss /= len(train_dataloader)
        return avg_batch_loss

    def init_centroid(self, feature_extractor:CustomFeatureExtractor, data_loader:DataLoader):
        """
        This method initializes the memory bank points

        Args:
            feature_extractor (CustomFeatureExtractor) : feature extractor to be used
            training_dataloader (torch.utils.data.DataLoader) : training dataloader
        """

        memory_bank = 0

        for i, x in enumerate(tqdm(data_loader)):
            x = x.to(self.device)
            p = feature_extractor(x)

            if isinstance(p, dict):
                p = list(p.values())

            self.scale = p[0].size(2)
            phi_p = self.Descriptor(p)
            memory_bank = ((memory_bank * i) + torch.mean(phi_p, dim=0, keepdim=True).detach()) / (i+1)

        return memory_bank

    def get_model_size(self):
        feature_extractor_params = self.feature_extractor.get_size()
        descriptor_params = params_to_mb(count_params(self.Descriptor))
        memory_bank_params = params_to_mb(count_params(self.memory_bank))

        return {
            "feature_extractor": feature_extractor_params,
            "descriptor": descriptor_params,
            "memory_bank": memory_bank_params,
            "total": feature_extractor_params + descriptor_params + memory_bank_params
        }

    def load_model(self, path):

        """
        Load the CFA memory bank

        Parameters:
        ----------
            path (str): where the pt file containing the memory bank is stored
        """

        state_dict = torch.load(path)

        if "C" not in state_dict.keys():
            raise RuntimeError("Memory Bank tensor not in model checkpoint")

        # load the memory bank
        self.memory_bank = state_dict["C"]

        # load the Patch Descriptor
        self.Descriptor = Descriptor(self.gamma_d, int(state_dict["feature_maps_channels"]), self.backbone, self.device)
        desc_dict = {
            'layer.weight'  : state_dict['Descriptor.layer.weight'],
            'layer.bias' : state_dict['Descriptor.layer.bias'],
            'layer.conv.weight' : state_dict['Descriptor.layer.conv.weight'],
            'layer.conv.bias' : state_dict['Descriptor.layer.conv.bias']
        }
        self.Descriptor.load_state_dict(desc_dict)

        self.feature_maps_channels = nn.Parameter(state_dict["feature_maps_channels"], requires_grad=False)

    def save_anomaly_map(self, dirpath, anomaly_map, pred_score, filepath, x_type, mask):
        """
        Args:
            dirpath     (str)       : Output directory path.
            anomaly_map (np.ndarray): Anomaly map with the same size as the input image.
            filepath    (str)       : Path of the input image.
            x_type      (str)       : Anomaly type (e.g. "good", "crack", etc).
            mask
        """

        def cvt2heatmap(gray):
            return cv.applyColorMap(np.uint8(gray), cv.COLORMAP_JET)

        # Get the image file name.
        filename = os.path.basename(filepath)

        # Load the image file and resize.
        original_image = cv.imread(filepath)
        original_image = cv.resize(original_image, anomaly_map.shape[:2])

        # Normalize anomaly map for easier visualization.
        anomaly_map_norm = cvt2heatmap(255 * CFA.rescale(anomaly_map))

        # Overlay the anomaly map to the origimal image.
        output_image = (anomaly_map_norm / 2 + original_image / 2).astype(np.uint8)

        # Create a figure and axes
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        #convert the images to RGB
        original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        output_image = cv.cvtColor(output_image, cv.COLOR_BGR2RGB)

        # Display the input image
        axes[0].imshow(original_image)
        axes[0].set_title(f'Original Image {x_type}')
        axes[0].axis('off')

        # Display the mask image
        axes[1].imshow(mask.squeeze(), cmap ='gray')
        axes[1].set_title(f'Mask')
        axes[1].axis('off')

        # Display the final image
        axes[2].imshow(output_image)
        axes[2].set_title(f'Heatmap {pred_score}')
        axes[2].axis('off')

        # Show the plot
        plt.savefig(str(dirpath / f"{x_type}_{filename}.jpg"))


    # --------------- SEGMENTATION MASK PRODUCTION ---------------- #

    @staticmethod
    def upsample(x, size, mode):
        return (
            F.interpolate(x.unsqueeze(1), size=size, mode=mode, align_corners=False)
            .squeeze()
        )

    @staticmethod
    def gaussian_smooth(x, sigma=4):
        bs = x.shape[0]
        for i in range(0, bs):
            x[i] = gaussian_filter(x[i], sigma=sigma)
        return x

    @staticmethod
    def gaussian_smooth_torch(x, sigma=4):
        if x.ndim == 2:
            x = x.unsqueeze(0) # if processing (working on) one image

        blur = GaussianBlur(3, sigma)
        return blur(x)

    @staticmethod
    def rescale(x):
        # A heatmap whose values are all identical makes max() - min() == 0,
        # turning every value into NaN instead of a valid [0, 1] map.
        span = x.max() - x.min()
        return (x - x.min()) / span if span != 0 else torch.zeros_like(x)

    @staticmethod
    def get_threshold(gt: np.ndarray, score: np.ndarray) -> float:
        """
        Calculate the segmentation threshold

        Args:
            gt (np.array)    : ground truth masks
            score (np.array) : predicted masks

        Returns:
            threshold (float) : segmentation threshold
        """

        gt_mask = np.asarray(gt)
        precision, recall, thresholds = precision_recall_curve(
            gt_mask.flatten(), score.flatten()
        )
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

        # consider the threshold with the highest f1 score
        threshold = thresholds[np.argmax(f1)]

        return threshold
