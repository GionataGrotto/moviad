from __future__ import annotations
import pathlib
import os

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

from moviad.models.audio.audio_vad_model import AudioVADModel
from moviad.models.audio.components.cfa.descriptor import Descriptor
from moviad.models.training_args import TrainingArgs
from moviad.utilities.audio.audio_feature_extractor import AudioFeatureExtractor
from moviad.utilities.get_sizes import *

class CFA(AudioVADModel):

    def __init__(
            self,
            feature_extractor: AudioFeatureExtractor,
            backbone: str,
            device: torch.device,
            gamma_c:int = 1,
            gamma_d:int = 1,
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

        super().__init__(feature_extractor=feature_extractor, device=device)

        self.C   = 0
        self.nu = 1e-3

        self.gamma_c = gamma_c
        self.gamma_d = gamma_d
        self.alpha = 1e-1
        self.K = 3
        self.J = 3
        self.r   = nn.Parameter(1e-5*torch.ones(1), requires_grad=True)

        self.Descriptor = None
        self.backbone = backbone

        self.feature_maps_shape = None

    def train_epoch(
        self,
        epoch: int,
        train_dataloader: DataLoader,
        training_args: TrainingArgs,
    ):
        if self.Descriptor is None or not isinstance(self.C, nn.Parameter):
            self.initialize_memory_bank(train_dataloader)
        return super().train_epoch(epoch, train_dataloader, training_args)

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):
        if training_args.optimizer is None:
            training_args.optimizer = AdamW(
                [{"params": self.parameters()}],
                lr=1e-3,
                weight_decay=5e-4,
                amsgrad=True,
            )

        batch = self.batch_input(batch).to(self.device)
        training_args.optimizer.zero_grad()
        loss = self(batch)
        loss.backward()
        training_args.optimizer.step()
        return loss.item() if hasattr(loss, "item") else float(loss)

    def initialize_memory_bank(self, training_dataloader: DataLoader):
        """
        Initialize the memory bank

        Args:
            training_dataloader (DataLoader) : training dataloader
        """

        self.init_centroid(training_dataloader)
        self.C = rearrange(self.C, 'b c h w -> (b h w) c').detach()

        if self.gamma_c > 1:
            self.C = self.C.cpu().detach().numpy()
            self.C = KMeans(n_clusters=(self.h * self.w)//self.gamma_c, max_iter=3000).fit(self.C).cluster_centers_
            self.C = torch.Tensor(self.C).to(self.device)

        self.C = self.C.transpose(-1, -2).detach()
        self.C = nn.Parameter(self.C, requires_grad=False)

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
            self.feature_maps_shape = (1, int(self.feature_maps_channels.item()), p[0].shape[2], p[0].shape[3])

        phi_p = self.Descriptor(p)
        phi_p = rearrange(phi_p, 'b c h w -> b (h w) c')

        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers  = torch.sum(torch.pow(self.C, 2), 0, keepdim=True)
        f_c      = 2 * torch.matmul(phi_p, (self.C))
        dist     = features + centers - f_c
        dist     = torch.sqrt(dist)

        n_neighbors = self.K
        dist     = dist.topk(n_neighbors, largest=False).values

        dist = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
        dist = dist.unsqueeze(-1)

        if self.training:
            return self.soft_boundary(phi_p)
        else:
            h = p[0].size(2)
            w = p[0].size(3)
            scores = rearrange(dist, 'b (h w) c -> b c h w', h=h, w=w).cpu().detach()

            heatmaps = torch.mean(scores, dim=1)
            heatmaps = CFA.upsample(heatmaps, size=self.feature_extractor.spec_shape[-2:], mode="bilinear")

            # added for robustness to 1 sample batches
            if len(heatmaps.shape) == 2:
                heatmaps = heatmaps.unsqueeze(0)

            heatmaps = CFA.gaussian_smooth_torch(heatmaps, sigma=4)
            img_scores = CFA.rescale(heatmaps)
            img_scores = scores.reshape(scores.shape[0], -1).max(axis=1).values

            tmp_scores = heatmaps.topk(5,dim=2).values.mean(dim=2)

            if len(heatmaps.shape) == 2:
                return heatmaps.view(1,1,heatmaps.shape[0], heatmaps.shape[1]), img_scores, tmp_scores
            else:
                return heatmaps.unsqueeze(dim=1), img_scores, tmp_scores

    def soft_boundary(self, phi_p: torch.Tensor) -> float:
        """
        Loss calculation

        Args:
            phi_p (torch.Tensor) : batch of transformed features

        Returns:
            loss (float) : CFA loss
        """


        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers  = torch.sum(torch.pow(self.C, 2), 0, keepdim=True)
        f_c      = 2 * torch.matmul(phi_p, (self.C))
        dist     = features + centers - f_c
        n_neighbors = self.K + self.J
        dist     = dist.topk(n_neighbors, largest=False).values

        score = (dist[:, : , :self.K] - self.r**2)
        L_att = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score))

        score = (self.r**2 - dist[:, : , self.J:])
        L_rep  = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score - self.alpha))

        loss = L_att + L_rep

        return loss

    def init_centroid(self, data_loader:DataLoader):
        """
        This method initializes the memory bank points

        Args:
            feature_extractor (CustomFeatureExtractor) : feature extractor to be used
            training_dataloader (torch.utils.data.DataLoader) : training dataloader
        """

        for i, x in enumerate(tqdm(data_loader)):
            x = x.to(self.device)
            p = self.feature_extractor(x)

            if isinstance(p, dict):
                p = list(p.values())

            if not self.Descriptor:
                self.feature_maps_channels = nn.Parameter(sum(tensor.size(1) for tensor in p) * torch.ones(1), requires_grad = False)
                self.Descriptor = Descriptor(self.gamma_d, int(self.feature_maps_channels.item()), self.backbone).to(self.device)

            self.h = p[0].size(2)
            self.w = p[0].size(3)
            phi_p = self.Descriptor(p)
            self.C = ((self.C * i) + torch.mean(phi_p, dim=0, keepdim=True).detach()) / (i+1)

    def get_model_size_and_macs(self) -> tuple[dict, float]:

        """
        This method returns the model size and inference MACs

        Returns:
            tuple:
                [0] : dict with all model components sizes, macs and number of parameters
                [1] : total size of the AD model
        """

        sizes = {}

        # get feature extractor and patch descriptor size, params and macs

        macs, params = get_model_macs(self.feature_extractor.model)
        sizes["feature_extractor"] = {
            "size" : get_torch_model_size(self.feature_extractor.model),
            "params" : params,
            "macs" : macs
        }

        macs, params = get_model_macs(self.Descriptor, self.feature_maps_shape)
        sizes["patch_descriptor"] = {
            "size" : get_torch_model_size(self.Descriptor),
            "params" : params,
            "macs" : macs
        }

        # get MB size and shape
        sizes["memory_bank"] = {
            "size" : get_tensor_size(self.C),
            "type" : str(self.C.dtype),
            "shape" : self.C.shape
        }

        total_size = sizes["feature_extractor"]["size"] + sizes["patch_descriptor"]["size"] + sizes["memory_bank"]["size"]

        return sizes, total_size

    def load_model(self, path = None, state_dict = None):

        """
        Load the CFA memory bank

        Parameters:
        ----------
            path (str): where the pt file containing the memory bank is stored
        """

        if not path and not state_dict:
            raise ValueError("Either path or state_dict must be provided")
        
        if path:
            state_dict = torch.load(path)

        if "C" not in state_dict.keys():
            raise RuntimeError("Memory Bank tensor not in model checkpoint")

        # load the memory bank
        self.C = state_dict["C"]

        # load the Patch Descriptor
        self.Descriptor = Descriptor(self.gamma_d, int(state_dict["feature_maps_channels"]), self.backbone)
        desc_dict = {
            'layer.weight'  : state_dict['Descriptor.layer.weight'],
            'layer.bias' : state_dict['Descriptor.layer.bias'],
            'layer.conv.weight' : state_dict['Descriptor.layer.conv.weight'],
            'layer.conv.bias' : state_dict['Descriptor.layer.conv.bias']
        }
        self.Descriptor.load_state_dict(desc_dict)

        self.feature_maps_channels = nn.Parameter(state_dict["feature_maps_channels"], requires_grad=False)

    def reset_model(self):
        self.C = 0
        self.Descriptor = None
        self.feature_maps_shape = None

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
        blur = GaussianBlur(3, sigma)
        return blur(x)

    @staticmethod
    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())

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
