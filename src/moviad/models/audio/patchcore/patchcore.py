"""PyTorch model for the audio PatchCore implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from itertools import islice
from typing import Union

import torch
from torch import Tensor
from tqdm import tqdm

from moviad.models.audio.audio_vad_model import AudioVADModel
from moviad.models.audio.components.band_scores import (
    BandScoreStats,
    holdout_period,
    is_holdout,
    standardize_by_band,
)
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
    DISTANCE_CHUNK_ELEMENTS = 2**24

    def __init__(
        self,
        device: torch.device,
        input_size: tuple[int],
        feature_extractor: Union[CustomFeatureExtractor, AudioFeatureExtractor],
        num_neighbors: int = 9,
        memory_bank_size: int = 30000,
        blur: bool = True,
        normalize_maps: bool = False,
        standardize_features: bool = False,
        band_banks: bool = False,
        score_standardization: str = "none",
        holdout_fraction: float = 0.1,
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
            standardize_features: z-score every channel of the fused features with the mean and
                standard deviation of the training set. Without it the layers with large
                activations dominate the euclidean distances.
            band_banks: one memory bank per frequency position of the feature map instead of a
                single one (``memory_bank_size`` is split evenly). A patch is then compared only
                with normal patches of its own band; time stays free, as it should for audio.
            score_standardization: ``"band"`` standardises the patch scores of every frequency band
                with the mean / std of held-out normal clips (``holdout_fraction`` of the training
                batches, excluded from the memory bank); ``"none"`` leaves the scores untouched.
        """
        super().__init__(
            feature_extractor=feature_extractor,
            input_size=input_size,
            device=device,
        )
        if len(input_size) != 2:
            raise ValueError("input_size should be a tuple of 2 integers")
        if score_standardization not in ("none", "band"):
            raise ValueError(f"score_standardization must be 'none' or 'band', got {score_standardization!r}")

        self.num_neighbors = num_neighbors
        self.memory_bank_size = memory_bank_size
        self.feature_pooler = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.anomaly_map_generator = AnomalyMapGenerator(blur=blur, normalize=normalize_maps)

        self.standardize_features = standardize_features
        self.band_banks = band_banks
        self.score_standardization = score_standardization
        self.holdout_fraction = holdout_fraction
        self.register_buffer("memory_bank", Tensor())
        self.register_buffer("feature_mean", Tensor())
        self.register_buffer("feature_std", Tensor())
        # the per-band banks are stored concatenated in memory_bank; this holds their sizes
        self.register_buffer("band_bank_sizes", torch.zeros(0, dtype=torch.long))
        self.register_buffer("score_mean", Tensor())
        self.register_buffer("score_std", Tensor())
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
        try:
            num_batches = len(train_dataloader)
        except TypeError:
            num_batches = None

        if self.standardize_features:
            self.fit_standardizer(train_dataloader)

        standardize_scores = self.score_standardization == "band"
        period = holdout_period(self.holdout_fraction) if standardize_scores else None
        # a bank needs data: the held-out batches do not contribute to it
        bank_batches = num_batches if not (num_batches and period) else max(1, num_batches - num_batches // period)

        coresets: list[StreamingCoreset] | None = None  # one per band, created on the first batch
        global_coreset = None if self.band_banks else StreamingCoreset(self.memory_bank_size, projection_dim, seed)
        held_out: list[Tensor] = []

        for index, batch in enumerate(tqdm(train_dataloader, desc="PatchCore embedding extraction")):
            embedding_map = self.embed(self.batch_input(batch)).detach()
            if period and is_holdout(index, period):
                held_out.append(embedding_map.cpu())
                continue
            batch_size, channels, height, width = embedding_map.shape
            if self.band_banks:
                if coresets is None:
                    band_size = self.memory_bank_size // width
                    if band_size < 1:
                        raise ValueError(f"memory_bank_size {self.memory_bank_size} is too small for {width} bands")
                    coresets = [StreamingCoreset(band_size, projection_dim, seed + band) for band in range(width)]
                for band, coreset in enumerate(coresets):
                    band_embedding = embedding_map[:, :, :, band].permute(0, 2, 1).reshape(-1, channels)
                    keep = math.ceil(pool_factor * coreset.bank_size / bank_batches) if bank_batches else math.ceil(0.1 * band_embedding.shape[0])
                    coreset.add(band_embedding.to(selection_device), keep)
            else:
                embedding = self.reshape_embedding(embedding_map).to(selection_device)
                keep = math.ceil(pool_factor * self.memory_bank_size / bank_batches) if bank_batches else math.ceil(0.1 * embedding.shape[0])
                global_coreset.add(embedding, keep)

        if self.band_banks:
            if coresets is None:
                raise RuntimeError("Cannot build a memory bank from an empty dataset")
            banks = [coreset.result(selection_device) for coreset in coresets]
            self.band_bank_sizes = torch.tensor([bank.shape[0] for bank in banks], dtype=torch.long)
            self.memory_bank = torch.cat(banks).to(self.device)
        else:
            self.band_bank_sizes = torch.zeros(0, dtype=torch.long)
            self.memory_bank = global_coreset.result(selection_device).to(self.device)

        if standardize_scores:
            self._fit_score_statistics(held_out)
        else:
            self.score_mean, self.score_std = Tensor(), Tensor()

    @torch.no_grad()
    def _fit_score_statistics(self, held_out: list[Tensor]) -> None:
        """Per-band mean / std of the patch scores of normal clips that are not in the memory bank."""
        statistics = BandScoreStats()
        for embedding_map in held_out:
            scores, _ = self._patch_scores(embedding_map.to(self.device))
            statistics.update(scores)
        mean, std = statistics.finalize()
        self.score_mean, self.score_std = mean.to(self.device), std.to(self.device)

    @torch.no_grad()
    def fit_standardizer(self, train_dataloader, max_batches: int = 16) -> None:
        """Per-channel mean / std of the fused features over the first ``max_batches`` batches."""
        self.feature_mean = Tensor()  # extract raw features, not yet standardised
        total, square_total, count = 0.0, 0.0, 0
        for batch in islice(train_dataloader, max_batches):
            features = self.embed(self.batch_input(batch)).double()
            total = total + features.sum(dim=(0, 2, 3))
            square_total = square_total + (features * features).sum(dim=(0, 2, 3))
            count += features.shape[0] * features.shape[2] * features.shape[3]
        if count == 0:
            raise RuntimeError("Cannot standardise the features of an empty dataset")
        mean = total / count
        std = (square_total / count - mean * mean).clamp_min(0).sqrt().clamp_min(1e-6)
        self.feature_mean = mean.float().to(self.device)
        self.feature_std = std.float().to(self.device)

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
        fused = fuse_feature_maps(features)
        if self.standardize_features and self.feature_mean.numel():
            fused = (fused - self.feature_mean.view(1, -1, 1, 1)) / self.feature_std.view(1, -1, 1, 1)
        return fused

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

        scores, locations = self._patch_scores(embedding_map)  # (B, H, W)
        if self.score_mean.numel():
            scores = standardize_by_band(scores, self.score_mean, self.score_std)
        patch_scores = scores.reshape(batch_size, -1)
        locations = locations.reshape(batch_size, -1)
        pred_scores = self.compute_anomaly_score(patch_scores, locations, embedding, width=width)

        anomaly_maps = self.anomaly_map_generator(
            patch_scores.reshape(batch_size, 1, height, width), image_size=self.input_size
        )
        # Per time frame score: mean of the five strongest frequency responses.
        return anomaly_maps, pred_scores, temporal_topk_scores(anomaly_maps)

    def _band_offsets(self) -> list[int]:
        sizes = self.band_bank_sizes.tolist()
        return [sum(sizes[:band]) for band in range(len(sizes) + 1)]

    def _patch_scores(self, embedding_map: Tensor) -> tuple[Tensor, Tensor]:
        """Distance of every patch to its nearest memory-bank patch and the bank index, both ``(B, H, W)``."""
        batch_size, channels, height, width = embedding_map.shape
        if self.band_bank_sizes.numel() == 0:
            scores, locations = self.nearest_neighbors(self.reshape_embedding(embedding_map), n_neighbors=1)
            return scores.reshape(batch_size, height, width), locations.reshape(batch_size, height, width)

        if width != self.band_bank_sizes.numel():
            raise ValueError(
                f"The memory banks were built for {self.band_bank_sizes.numel()} frequency bands "
                f"but the feature map has {width}"
            )
        offsets = self._band_offsets()
        scores = embedding_map.new_empty(batch_size, height, width)
        locations = torch.empty(batch_size, height, width, dtype=torch.long, device=embedding_map.device)
        for band in range(width):
            queries = embedding_map[:, :, :, band].permute(0, 2, 1).reshape(-1, channels)
            bank = self.memory_bank[offsets[band] : offsets[band + 1]]
            band_scores, band_locations = self._search(queries, bank, 1)
            scores[:, :, band] = band_scores.reshape(batch_size, height)
            locations[:, :, band] = (band_locations + offsets[band]).reshape(batch_size, height)
        return scores, locations

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        """``[B, C, H, W]`` -> ``[B*H*W, C]``."""
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding.size(1))

    @staticmethod
    def euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
        return torch.cdist(x, y)

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int) -> tuple[Tensor, Tensor]:
        """Distances and memory-bank indices of the nearest neighbours of each row (single bank)."""
        return self._search(embedding, self.memory_bank, n_neighbors)

    def _search(self, embedding: Tensor, bank: Tensor, n_neighbors: int) -> tuple[Tensor, Tensor]:
        """Nearest neighbours of each row of ``embedding`` in ``bank``.

        The query is processed in chunks so that the distance matrix never exceeds
        ``DISTANCE_CHUNK_ELEMENTS`` entries, however many patches a batch has. If the GPU is
        shared and even that does not fit, the chunk is halved until it does.
        """
        rows = max(1, self.DISTANCE_CHUNK_ELEMENTS // max(bank.shape[0], 1))
        scores, locations = [], []
        start = 0
        while start < embedding.shape[0]:
            try:
                distances = self.euclidean_distance(embedding[start : start + rows], bank)
                if n_neighbors == 1:
                    score, location = distances.min(dim=1)
                else:
                    score, location = distances.topk(k=n_neighbors, largest=False, dim=1)
            except torch.cuda.OutOfMemoryError:
                if rows == 1:
                    raise
                distances = None
                torch.cuda.empty_cache()
                rows = max(1, rows // 2)
                self.DISTANCE_CHUNK_ELEMENTS = rows * max(bank.shape[0], 1)  # remember for the next calls
                continue
            scores.append(score)
            locations.append(location)
            start += rows
        return torch.cat(scores), torch.cat(locations)

    def _support_samples(self, nearest_sample: Tensor, neighbors: int, bands: Tensor | None) -> Tensor:
        """Memory-bank indices of the ``neighbors`` nearest samples of each ``nearest_sample`` row.

        With per-band banks the search stays inside the band of the patch that produced the sample.
        """
        if bands is None:
            return self.nearest_neighbors(nearest_sample, n_neighbors=neighbors)[1]
        offsets = self._band_offsets()
        neighbors = min(neighbors, int(self.band_bank_sizes.min()))
        support = torch.empty(nearest_sample.shape[0], neighbors, dtype=torch.long, device=nearest_sample.device)
        for band in bands.unique().tolist():
            rows = (bands == band).nonzero(as_tuple=True)[0]
            bank = self.memory_bank[offsets[band] : offsets[band + 1]]
            support[rows] = self._search(nearest_sample[rows], bank, neighbors)[1] + offsets[band]
        return support

    def compute_anomaly_score(
        self, patch_scores: Tensor, locations: Tensor, embedding: Tensor, width: int | None = None
    ) -> Tensor:
        """Image-level score of PatchCore (paper, section 3.3: re-weighting).

        ``width`` is the number of frequency bands of the feature map; it is only needed with
        per-band banks, to find the band of the patch that gives the image score.
        """
        smallest_bank = int(self.band_bank_sizes.min()) if self.band_bank_sizes.numel() else self.memory_bank.shape[0]
        neighbors = min(self.num_neighbors, smallest_bank)
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
        bands = None if self.band_bank_sizes.numel() == 0 else max_patches % width
        support = self._support_samples(nearest_sample, neighbors, bands)
        # 3. distance of the test patch to the support samples, softmax -> weight
        support_distances = torch.linalg.vector_norm(
            max_patch_features.unsqueeze(1) - self.memory_bank[support], dim=-1
        )
        weights = 1 - torch.softmax(support_distances, dim=1)[:, 0]
        return weights * score

    # ---------------------------------------------------------- persistence
    _SIZED_BUFFERS = ("memory_bank", "feature_mean", "feature_std", "band_bank_sizes", "score_mean", "score_std")
    #: buffers that older checkpoints do not have
    _OPTIONAL_BUFFERS = ("band_bank_sizes", "score_mean", "score_std")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # The buffers are registered empty; a fitted checkpoint has other shapes.
        for name in self._SIZED_BUFFERS:
            key = prefix + name
            if key in state_dict and state_dict[key].shape != getattr(self, name).shape:
                setattr(self, name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        optional = [prefix + name for name in self._OPTIONAL_BUFFERS]
        if not self.standardize_features:  # a checkpoint from before the feature standardiser existed
            optional += [prefix + "feature_mean", prefix + "feature_std"]
        missing_keys[:] = [key for key in missing_keys if key not in optional]

    def load_model(self, path):
        """Load a memory bank saved with ``save`` (or a bare ``{"memory_bank": tensor}`` dict)."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        if "memory_bank" not in state_dict:
            raise RuntimeError("Memory Bank tensor not in model checkpoint")
        self.memory_bank = state_dict["memory_bank"].to(self.device)

    def reset_model(self):
        self.memory_bank = Tensor()
        self.band_bank_sizes = torch.zeros(0, dtype=torch.long)
        self.score_mean, self.score_std = Tensor(), Tensor()

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
