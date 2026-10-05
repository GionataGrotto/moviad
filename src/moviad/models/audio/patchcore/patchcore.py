"""PyTorch model for the audio PatchCore implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor
from tqdm import tqdm

from moviad.models.audio.audio_vad_model import AudioVADModel
from moviad.models.audio.components.feature_ops import (
    fuse_feature_maps,
    temporal_topk_scores,
)
from moviad.models.audio.patchcore.anomaly_map import AnomalyMapGenerator
from moviad.models.audio.patchcore.coreset import StreamingCoreset
from moviad.models.training_args import TrainingArgs
from moviad.utilities.audio.audio_feature_extractor import AudioFeatureExtractor
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor


class PatchCore(AudioVADModel):
    """PatchCore with a memory-bounded fit and a memory-bounded nearest-neighbour search."""

    #: Upper bound on the size of one query-vs-memory-bank distance matrix.
    DISTANCE_CHUNK_ELEMENTS = 2**26

    def __init__(
        self,
        device: torch.device,
        input_size: tuple[int],
        feature_extractor: Union[CustomFeatureExtractor, AudioFeatureExtractor],
        num_neighbors: int = 9,
        memory_bank_size: int = 30000,
        blur: bool = True,
        normalize_maps: bool = False,
    ) -> None:
        """
        Args:
            device: device used for feature extraction and nearest-neighbour search.
            input_size: (height, width) the anomaly map is upsampled to.
            feature_extractor: backbone returning a list/dict of feature maps.
            num_neighbors: neighbours used by the PatchCore re-weighting of the image score.
            memory_bank_size: number of patch embeddings kept after coreset selection.
            blur: Gaussian smoothing of the anomaly map.
            normalize_maps: min-max normalise maps per batch. Leave off for any use
                where scores of different batches are compared (see ``AnomalyMapGenerator``).
        """
        super().__init__(
            feature_extractor=feature_extractor,
            input_size=input_size,
            device=device,
        )
        if len(input_size) != 2:
            raise ValueError("input_size should be a tuple of 2 integers")

        self.num_neighbors = num_neighbors
        self.memory_bank_size = memory_bank_size
        self.feature_pooler = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.anomaly_map_generator = AnomalyMapGenerator(blur=blur, normalize=normalize_maps)

        self.register_buffer("memory_bank", Tensor())
        self.memory_bank: Tensor

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        train_dataloader,
        force_cpu: bool = False,
        pool_factor: int = 4,
        projection_dim: int = 128,
        seed: int = 0,
    ) -> None:
        """Build the memory bank from ``train_dataloader`` with bounded memory.

        Each batch is reduced to a few representatives by k-center greedy while it
        is streamed; the pooled representatives (about ``pool_factor`` times the
        bank size) are then reduced to ``memory_bank_size``.

        Args:
            force_cpu: run the greedy selection on CPU (useful when the GPU is full).
            pool_factor: size of the intermediate pool, in multiples of the bank.
            projection_dim: random-projection size used only for greedy distances.
        """
        selection_device = (
            torch.device("cpu") if force_cpu or self.device.type == "cpu" else self.device
        )
        coreset = StreamingCoreset(self.memory_bank_size, projection_dim, seed)
        try:
            num_batches = len(train_dataloader)
        except TypeError:
            num_batches = None

        was_training = self.training
        self.train()
        try:
            for batch in tqdm(train_dataloader, desc="PatchCore embedding extraction"):
                embedding = self(self.batch_input(batch)).detach().to(selection_device)
                if num_batches:
                    keep = math.ceil(pool_factor * self.memory_bank_size / num_batches)
                else:
                    keep = math.ceil(0.1 * embedding.shape[0])
                coreset.add(embedding, keep)
        finally:
            self.train(was_training)

        self.memory_bank = coreset.result(selection_device).to(self.device)

    def train_epoch(self, epoch, train_dataloader, training_args: TrainingArgs):
        self.fit(train_dataloader, force_cpu=getattr(training_args, "force_cpu", False))
        return 0.0

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):
        raise NotImplementedError("Audio PatchCore is fitted with fit/train_epoch, not train_step")

    # -------------------------------------------------------------- forward
    def embed(self, input_tensor: Tensor) -> Tensor:
        """Locally-aware patch features ``(B, C, H, W)`` for a batch of waveforms."""
        with torch.no_grad():
            features = self.feature_extractor(input_tensor.to(self.device))
        if isinstance(features, dict):
            features = list(features.values())
        features = [self.feature_pooler(feature) for feature in features]
        # TODO: audio positional encoding should only separate frequency bands, not time.
        return fuse_feature_maps(features)

    def forward(self, input_tensor: Tensor) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Patch embeddings in training mode, ``(anomaly_maps, scores, temporal_scores)`` in eval mode."""
        embedding_map = self.embed(input_tensor)
        batch_size, _, height, width = embedding_map.shape
        embedding = self.reshape_embedding(embedding_map)  # (B*H*W, C)

        if self.training:
            return embedding

        if self.memory_bank.numel() == 0:
            raise RuntimeError("The memory bank is empty: fit the model before evaluating it")
        self.memory_bank = self.memory_bank.to(self.device)

        patch_scores, locations = self.nearest_neighbors(embedding, n_neighbors=1)
        patch_scores = patch_scores.reshape(batch_size, -1)
        locations = locations.reshape(batch_size, -1)
        pred_scores = self.compute_anomaly_score(patch_scores, locations, embedding)

        anomaly_maps = self.anomaly_map_generator(
            patch_scores.reshape(batch_size, 1, height, width), image_size=self.input_size
        )
        # Per time frame score: mean of the five strongest frequency responses.
        return anomaly_maps, pred_scores, temporal_topk_scores(anomaly_maps)

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        """``[B, C, H, W]`` -> ``[B*H*W, C]``."""
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding.size(1))

    @staticmethod
    def euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
        return torch.cdist(x, y)

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int) -> tuple[Tensor, Tensor]:
        """Distances and memory-bank indices of the nearest neighbours of each row.

        The query is processed in chunks so that the distance matrix never exceeds
        ``DISTANCE_CHUNK_ELEMENTS`` entries, however many patches a batch has.
        """
        bank = self.memory_bank
        rows = max(1, self.DISTANCE_CHUNK_ELEMENTS // max(bank.shape[0], 1))
        scores, locations = [], []
        for chunk in embedding.split(rows):
            distances = self.euclidean_distance(chunk, bank)
            if n_neighbors == 1:
                score, location = distances.min(dim=1)
            else:
                score, location = distances.topk(k=n_neighbors, largest=False, dim=1)
            scores.append(score)
            locations.append(location)
        return torch.cat(scores), torch.cat(locations)

    def compute_anomaly_score(self, patch_scores: Tensor, locations: Tensor, embedding: Tensor) -> Tensor:
        """Image-level score of PatchCore (paper, section 3.3: re-weighting)."""
        neighbors = min(self.num_neighbors, self.memory_bank.shape[0])
        if neighbors <= 1:
            return patch_scores.amax(dim=1)

        batch_size, num_patches = patch_scores.shape
        rows = torch.arange(batch_size, device=patch_scores.device)
        # 1. the patch farthest from its nearest neighbour (m^test,*) and its score s^*
        max_patches = patch_scores.argmax(dim=1)
        max_patch_features = embedding.reshape(batch_size, num_patches, -1)[rows, max_patches]
        score = patch_scores[rows, max_patches]
        # 2. its nearest neighbour in the bank (m^*) and the neighbours of that neighbour
        nearest_sample = self.memory_bank[locations[rows, max_patches]]
        _, support = self.nearest_neighbors(nearest_sample, n_neighbors=neighbors)
        # 3. distance of the test patch to the support samples, softmax -> weight
        support_distances = torch.linalg.vector_norm(
            max_patch_features.unsqueeze(1) - self.memory_bank[support], dim=-1
        )
        weights = 1 - torch.softmax(support_distances, dim=1)[:, 0]
        return weights * score

    # ---------------------------------------------------------- persistence
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # The buffer is registered empty; a fitted checkpoint has another shape.
        key = prefix + "memory_bank"
        if key in state_dict and state_dict[key].shape != self.memory_bank.shape:
            self.memory_bank = torch.empty_like(state_dict[key])
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def load_model(self, path):
        """Load a memory bank saved with ``save`` (or a bare ``{"memory_bank": tensor}`` dict)."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        if "memory_bank" not in state_dict:
            raise RuntimeError("Memory Bank tensor not in model checkpoint")
        self.memory_bank = state_dict["memory_bank"].to(self.device)

    def reset_model(self):
        self.memory_bank = Tensor()

    def get_model_size_and_macs(self):
        from moviad.utilities.get_sizes import (
            get_model_macs,
            get_tensor_size,
            get_torch_model_size,
        )

        macs, params = get_model_macs(self.feature_extractor.model)
        sizes = {
            "feature_extractor": {
                "size": get_torch_model_size(self.feature_extractor.model),
                "params": params,
                "macs": macs,
            },
            "memory_bank": {
                "size": get_tensor_size(self.memory_bank),
                "type": str(self.memory_bank.dtype),
                "shape": self.memory_bank.shape,
            },
        }
        return sizes, sizes["feature_extractor"]["size"] + sizes["memory_bank"]["size"]
