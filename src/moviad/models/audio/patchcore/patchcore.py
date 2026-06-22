"""PyTorch model for the PatchCore model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from moviad.models.audio.audio_vad_model import AudioVADModel
from moviad.models.training_args import TrainingArgs
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.audio.audio_feature_exctractor import AudioFeatureExtractor
from moviad.models.audio.patchcore.anomaly_map import AnomalyMapGenerator
from moviad.models.audio.patchcore.kcenter_greedy import KCenterGreedy


class PatchCore(AudioVADModel):
    """Patchcore Module."""

    def __init__(
        self,
        device: torch.device,
        input_size: tuple[int],
        feature_extractor: Union[CustomFeatureExtractor, AudioFeatureExtractor],
        num_neighbors: int = 9,
        memory_bank_size: int = 30000,
        blur: True = True,
    ) -> None:
        """
        Constructor of the patch-core model

        Args:
            device (torch.device): device to be used during the training
            input_size (tuple[int]): size of the input images
            feature_extractor: feature extractor to be used
            num_neighbors (int): number of neighbors to be considered in the k-nn search
        """

        super().__init__(
            feature_extractor=feature_extractor,
            input_size=input_size,
            device=device,
        )

        self.num_neighbors = num_neighbors
        assert len(input_size) == 2, "Input size should be a tuple of 2 integers"
        self.memory_bank_size = memory_bank_size

        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator(blur=blur)

        self.register_buffer("memory_bank", Tensor())
        self.memory_bank: Tensor

    def train_epoch(
        self,
        epoch,
        train_dataloader,
        training_args: TrainingArgs,
    ):
        embeddings = []

        self.train()
        with torch.no_grad():
            print("Embedding Extraction:")
            for batch in tqdm(iter(train_dataloader)):
                batch = self.batch_input(batch).to(self.device)
                embeddings.append(self(batch))

            embeddings = torch.cat(embeddings, dim=0)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            print("Coreset Extraction:")
            sampler = KCenterGreedy(embeddings, self.device)
            sampled_idxs = sampler.get_coreset_idx_randomp(
                embeddings.cpu(),
                memory_bank_size=self.memory_bank_size,
                force_cpu=getattr(training_args, "force_cpu", False),
            )
            self.memory_bank = embeddings[sampled_idxs]

        return 0.0

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):
        raise NotImplementedError("Audio PatchCore is fitted with train_epoch, not train_step")

    def forward(self, input_tensor: Tensor) -> Tensor | dict[str, Tensor]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.

        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode, return the embedding in training mode

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Tensor | dict[str, Tensor]: Embedding for training,
                anomaly map and anomaly score for testing.
        """

        # extract the features for the input tensor
        with torch.no_grad():
            features = self.feature_extractor(input_tensor.to(self.device))

        # concatenate the embeddings
        if isinstance(features, dict):
            features = list(features.values())

        # Apply smoothing (3x3 average pooling) to the features.
        smoothing = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        features = [smoothing(feature) for feature in features]

        # Compute maximum shape.
        H_max = max([f.shape[2] for f in features])
        W_max = max([f.shape[3] for f in features])

        # Create resize function instance.
        resizer = torch.nn.Upsample(size=(H_max, W_max), mode="nearest")

        # Apply resize function for all input tensors.
        features = [resizer(f) for f in features]

        embedding = torch.cat(features, dim=1)

        # TODO: add positional encoding here, to the embedding vectors
        # NOTE: for audio, the encoding should only discrimiate between
        #   patches of different frequency bands and not on time

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        # embedding shape: (#num_patches, emb_dim)

        if self.training:
            output = embedding
        else:
            self.memory_bank.to(self.device)

            # apply nearest neighbor search
            patch_scores, locations = self.nearest_neighbors(
                embedding=embedding, n_neighbors=1
            )

            # print(patch_scores.shape)
            # print("Locations" + str(locations.shape))

            # reshape to batch dimension
            patch_scores = patch_scores.reshape((batch_size, -1))
            locations = locations.reshape((batch_size, -1))

            # compute the anomaly score of the images
            pred_scores = self.compute_anomaly_score(patch_scores, locations, embedding)

            # reshape to w,h
            patch_scores = patch_scores.reshape((batch_size, 1, width, height))

            # get the anomaly map
            anomaly_maps = self.anomaly_map_generator(
                patch_scores, image_size=self.input_size
            )

            output = (anomaly_maps, pred_scores)

        return output

    def generate_embedding(self, features: dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features: dict[str:Tensor]: Hierarchical feature map from a CNN

        Returns:
            Embedding vector [Tensor]
        """

        embeddings = [features[self.layers[0]]]

        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(
                layer_embedding, size=embeddings[0].shape[-2], mode="bilinear"
            )
            embeddings.append(layer_embedding)

        embeddings = torch.cat(embeddings, 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        """
        Reshape Embedding.

        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """

        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding

    @staticmethod
    def euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
        """
        Calculates pair-wise distance between row vectors in x and those in y.

        Args:
            x: input tensor 1
            y: input tensor 2

        Returns:
            Matrix of distances between row vectors in x and y.
        """

        return torch.cdist(x, y)

    def nearest_neighbors(
        self, embedding: Tensor, n_neighbors: int
    ) -> tuple[Tensor, Tensor]:
        """
        Nearest neighbors using brute force method and euclidean norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """
        distances = PatchCore.euclidean_distance(embedding, self.memory_bank)

        if n_neighbors == 1:
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(
                k=n_neighbors, largest=False, dim=1
            )

        return patch_scores, locations

    def compute_anomaly_score(
        self, patch_scores: Tensor, locations: Tensor, embedding: Tensor
    ) -> Tensor:
        """
        Compute Image-Level Anomaly Score.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores

        Returns:
            Tensor: Image-level anomaly scores
        """

        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)

        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(
            patch_scores, dim=1
        )  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[
            torch.arange(batch_size), max_patches
        ]

        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[
            torch.arange(batch_size), max_patches
        ]  # indices of m^* in the paper

        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.memory_bank[nn_index, :]  # m^* in the paper

        # indices of N_b(m^*) in the paper
        memory_bank_effective_size = self.memory_bank.shape[
            0
        ]  # edge case when memory bank is too small
        _, support_samples = self.nearest_neighbors(
            nn_sample,
            n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
        )

        # 4. Find the distance of the patch features to each of the support samples
        distances = PatchCore.euclidean_distance(
            max_patches_features.unsqueeze(1),
            self.memory_bank[support_samples],
        )

        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]

        # 6. Apply the weight factor to the score
        return weights * score  # s in the paper

    def save_anomaly_map(
        self, dirpath, anomaly_map, pred_score, filepath, x_type, mask
    ):
        """
        Args:
            dirpath     (str)       : Output directory path.
            anomaly_map (np.ndarray): Anomaly map with the same size as the input image.
            filepath    (str)       : Path of the input image.
            x_type      (str)       : Anomaly type (e.g. "good", "crack", etc).
            contour     (float)     : Threshold of contour, or None.
        """

        def min_max_norm(image):
            a_min, a_max = image.min(), image.max()
            return (image - a_min) / (a_max - a_min)

        def cvt2heatmap(gray):
            return cv.applyColorMap(np.uint8(gray), cv.COLORMAP_JET)

        # Get the image file name.
        filename = os.path.basename(filepath)

        # Load the image file and resize.
        original_image = cv.imread(filepath)
        original_image = cv.resize(original_image, anomaly_map.shape[:2])

        # Normalize anomaly map for easier visualization.
        anomaly_map_norm = cvt2heatmap(255 * min_max_norm(anomaly_map))

        # Overlay the anomaly map to the origimal image.
        output_image = (anomaly_map_norm / 2 + original_image / 2).astype(np.uint8)

        # Create a figure and axes
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        # convert the images to RGB
        original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        output_image = cv.cvtColor(output_image, cv.COLOR_BGR2RGB)

        # Display the input image
        axes[0].imshow(original_image)
        axes[0].set_title(f"Original Image {x_type}")
        axes[0].axis("off")

        # Display the mask image
        axes[1].imshow(mask.squeeze(), cmap="gray")
        axes[1].set_title(f"Mask")
        axes[1].axis("off")

        # Display the final image
        axes[2].imshow(output_image)
        axes[2].set_title(f"Heatmap {pred_score}")
        axes[2].axis("off")

        # Show the plot
        plt.savefig(str(dirpath / f"{x_type}_{filename}.jpg"))

    def get_model_size_and_macs(self):
        from moviad.utilities.get_sizes import (
            get_model_macs,
            get_torch_model_size,
            get_tensor_size,
        )

        sizes = {}

        # get feature extractor size, params, and macs

        macs, params = get_model_macs(self.feature_extractor.model)
        sizes["feature_extractor"] = {
            "size": get_torch_model_size(self.feature_extractor.model),
            "params": params,
            "macs": macs,
        }

        # get MB size and shape
        sizes["memory_bank"] = {
            "size": get_tensor_size(self.memory_bank),
            "type": str(self.memory_bank.dtype),
            "shape": self.memory_bank.shape,
        }

        total_size = sizes["feature_extractor"]["size"] + sizes["memory_bank"]["size"]

        return sizes, total_size

    def load_model(self, path):
        """
        Load the Patchcore memory bank

        Parameters:
        ----------
            path (str): where the pt file containing the memory bank is stored
        """

        state_dict = torch.load(path, weights_only=False)

        if "memory_bank" not in state_dict.keys():
            raise RuntimeError("Memory Bank tensor not in model checkpoint")

        # load the memory bank
        self.memory_bank = state_dict["memory_bank"]

    def reset_model(self):
        self.memory_bank = Tensor()
