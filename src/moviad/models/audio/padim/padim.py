import os
import warnings
from typing import Any, Dict, List, Mapping, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from moviad.models.audio.audio_vad_model import AudioVADModel
from moviad.models.audio.components.feature_ops import (
    fuse_feature_maps,
    gaussian_smooth,
    temporal_topk_scores,
)
from moviad.models.training_args import TrainingArgs
from moviad.utilities.audio.audio_feature_extractor import AudioFeatureExtractor


# Dict: "backbone_model_name" -> {(layer_idxs): (true_dimension, random_projection_dimension)}
EMBEDDING_SIZES = {
    "phinet_1.2_0.5_6_downsampling": {
        (4, 5, 6): (200, 50),
        (5, 6, 7): (400, 100),
        (6, 7, 8): (576, 144),
        (2, 6, 7): (376, 94),
    },
    "micronet-m1": {
        (1, 2, 3): (40, 10),
        (2, 3, 4): (64, 16),
        (3, 4, 5): (112, 28),
        (2, 4, 5): (112, 28),
    },
    "mcunet-in3": {
        (3, 6, 9): (80, 20),
        (6, 9, 12): (112, 28),
        (9, 12, 15): (184, 46),
        (2, 6, 14): (136, 34),
    },
    "mobilenet_v2": {
        ("features.4", "features.7", "features.10"): (160, 40),
        ("features.7", "features.10", "features.13"): (224, 56),
        ("features.10", "features.13", "features.16"): (320, 80),
        ("features.3", "features.8", "features.14"): (248, 62),
    },
    "wide_resnet50_2": {("layer1", "layer2", "layer3"): (1792, 550)},
    "Cnn14": {("conv_block2", "conv_block3", "conv_block4"): (896, 225)},
    "HTSAT-base": {
        # HTSAT's 4 Swin stages have 256, 512, 1024, 1024 channels (the last
        # stage has no further PatchMerging, so it keeps stage 2's width).
        ("1", "2", "3"): (512 + 1024 + 1024, 225),
        ("0", "1", "2"): (256 + 512 + 1024, 225),
    },
}


def idx_to_layer_name(backbone_model_name, idx: Union[Tuple, List]):
    if backbone_model_name in ["wide_resnet50_2"]:
        return tuple(f"layer{i}" for i in idx)
    elif backbone_model_name == "mobilenet_v2":
        return tuple(f"features.{i}" for i in idx)
    else:
        return idx


class GaussianAccumulator:
    """Streaming per-patch mean / covariance in float64.

    Consumes ``(B, C, H, W)`` embedding batches and never holds more than one of
    them. The sums are taken on data shifted by the first batch mean, which keeps
    the single-pass covariance numerically stable, and the full-covariance update
    is one in-place batched matmul per batch (no temporaries the size of the
    covariance tensor).
    """

    def __init__(self, full_covariance: bool):
        self.full_covariance = full_covariance
        self.count = 0
        self.grid = None
        self.shift = None  # (P, C)
        self.sum = None  # (P, C)
        self.second_moment = None  # (P, C) or (P, C, C)

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor) -> None:
        batch, channels, height, width = embeddings.shape
        if self.grid is not None and self.grid != (height, width):
            raise ValueError(
                f"PaDiM fits one Gaussian per time-frequency patch, but the embedding grid "
                f"changed from {self.grid} to {(height, width)} during fitting. All training "
                "clips must have the same duration."
            )
        x = embeddings.double().reshape(batch, channels, height * width).permute(2, 0, 1)  # (P, B, C)
        if self.count == 0:
            self.grid = (height, width)
            self.shift = x.mean(dim=1)
            self.sum = torch.zeros_like(self.shift)
            shape = (x.shape[0], channels, channels) if self.full_covariance else self.shift.shape
            self.second_moment = torch.zeros(shape, dtype=x.dtype, device=x.device)
        x = x - self.shift.unsqueeze(1)
        self.sum += x.sum(dim=1)
        if self.full_covariance:
            self.second_moment.baddbmm_(x.transpose(1, 2), x)
        else:
            self.second_moment += (x * x).sum(dim=1)
        self.count += batch

    def finalize(self, covariance_reg: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("Cannot fit PaDiM on an empty dataset (or the accumulator was already finalized)")
        shifted_mean = self.sum / self.count
        mean = self.shift + shifted_mean
        denominator = max(self.count - 1, 1)
        covariance = self.second_moment
        if self.full_covariance:
            covariance -= self.count * shifted_mean.unsqueeze(2) * shifted_mean.unsqueeze(1)
            covariance /= denominator
            covariance.diagonal(dim1=1, dim2=2).add_(covariance_reg)
            covariance = covariance.float().permute(1, 2, 0)  # (C, C, P)
        else:
            covariance = (covariance - self.count * shifted_mean * shifted_mean) / denominator + covariance_reg
            covariance = covariance.float().transpose(0, 1)  # (C, P)
        self.count, self.second_moment = 0, None  # finalize consumes the buffers (it works in place)
        return mean.float().transpose(0, 1).cpu().numpy(), np.ascontiguousarray(covariance.cpu().numpy())


class Padim(AudioVADModel):

    HYPERPARAMS = [
        "class_name",
        "backbone_model_name",
        "t_d",
        "d",
        "gauss_mean",
        "gauss_cov",
        "diag_cov",
        "covariance_reg",
        "layers_idxs",
        "fit_grid",
    ]

    def __init__(
        self,
        backbone_model_name,
        class_name,
        device,
        layers_idxs: list,
        diag_cov=False,
        img_size=None,
        backbone_model=None,
        embedding_dim=None,
        covariance_reg=0.01,
    ):
        """
        Args:
            backbone_model_name: key of ``EMBEDDING_SIZES`` (e.g. 'Cnn14', 'HTSAT-base').
            class_name: label of the machine / class, used in checkpoint paths.
            layers_idxs: backbone layers whose feature maps are fused.
            diag_cov: keep only the per-patch variances instead of full covariance matrices.
            img_size: (height, width) the anomaly map is upsampled to; ``None`` keeps
                the input spectrogram size (or the embedding grid for waveform inputs).
            embedding_dim: size of the random channel subset (default from ``EMBEDDING_SIZES``).
        """
        super().__init__(
            feature_extractor=backbone_model,
            input_size=img_size,
            device=device,
        )
        self.class_name = class_name
        self.diag_cov = diag_cov
        self.covariance_reg = float(covariance_reg)
        self.backbone_model_name = backbone_model_name
        self.layers_idxs = layers_idxs
        self.backbone_model = self.load_backbone(backbone_model)
        self.feature_extractor = self.backbone_model

        # dimensionality reduction: random channel subset
        _, max_embedding_dim = self.lookup_embedding_size(backbone_model_name, layers_idxs)
        self.d = int(max_embedding_dim if embedding_dim is None else embedding_dim)
        if not 1 <= self.d <= self.t_d:
            raise ValueError(f"embedding_dim must be between 1 and {self.t_d}, got {self.d}")
        random_dims = torch.randperm(self.t_d)[: self.d]
        self.random_dimensions = torch.nn.Parameter(random_dims, requires_grad=False)

        self.img_size = img_size
        self._warned_grid_mismatch = False
        self.fit_grid = None
        self._gauss_mean = None
        self._gauss_cov = None
        self._inference_cache = {}

    # ------------------------------------------------------- fitted state
    @property
    def gauss_mean(self):
        return self._gauss_mean

    @gauss_mean.setter
    def gauss_mean(self, value):
        self._gauss_mean = value
        self._inference_cache = {}

    @property
    def gauss_cov(self):
        return self._gauss_cov

    @gauss_cov.setter
    def gauss_cov(self, value):
        self._gauss_cov = value
        self._inference_cache = {}

    def reset_model(self):
        self.gauss_mean = None
        self.gauss_cov = None
        self.fit_grid = None

    # ------------------------------------------------------------ backbone
    def load_backbone(self, backbone_model):
        """Create the frozen audio backbone if needed and look up the channel counts."""
        if backbone_model is None:
            backbone_model = AudioFeatureExtractor(
                model_name=self.backbone_model_name,
                layers_idx=self.layers_idxs,
                device=self.device,
                frozen=True,
            )
        self.t_d, default_d = self.lookup_embedding_size(self.backbone_model_name, self.layers_idxs)
        if not hasattr(self, "d"):
            self.d = default_d
        return backbone_model

    @staticmethod
    def lookup_embedding_size(backbone_model_name, layers_idxs):
        """Look up (total_channels, default_projection_dim) for a backbone/layer pair."""
        if backbone_model_name not in EMBEDDING_SIZES:
            raise KeyError(
                f"Unsupported backbone {backbone_model_name!r} for PaDiM. "
                f"Known backbones: {sorted(EMBEDDING_SIZES)}"
            )
        known_layers = EMBEDDING_SIZES[backbone_model_name]
        key = tuple(layers_idxs)
        if key not in known_layers:
            raise KeyError(
                f"Unsupported layer combination {key} for backbone "
                f"{backbone_model_name!r}. Known combinations: "
                f"{sorted(known_layers, key=str)}"
            )
        return known_layers[key]

    def _extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Feature maps of the configured layers, one tensor per layer."""
        with torch.no_grad():
            maps = self.backbone_model(x.to(self.device))
        maps = list(maps.values()) if isinstance(maps, dict) else list(maps)
        if len(maps) != len(self.layers_idxs):
            raise ValueError(
                f"The backbone returned {len(maps)} feature maps but PaDiM is configured "
                f"with {len(self.layers_idxs)} layers {tuple(self.layers_idxs)}"
            )
        return maps

    # ---------------------------------------------------------- embeddings
    @staticmethod
    def embedding_concat(x, y):
        """Concatenate two feature maps, replicating the coarser one over the finer grid."""
        return fuse_feature_maps([x, y])

    def _project(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.size(1) != self.t_d:
            raise ValueError(
                f"Expected {self.t_d} channels from layers {tuple(self.layers_idxs)} of "
                f"{self.backbone_model_name!r}, the backbone produced {fused.size(1)}"
            )
        return torch.index_select(fused, 1, self.random_dimensions.to(fused.device))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, d, H, W)`` projected embedding of a batch, kept on the model device."""
        return self._project(fuse_feature_maps(self._extract(x)))

    def raw_feature_maps_to_embeddings(self, layer_outputs: Dict[str, List[torch.Tensor]]):
        """Embedding of already extracted ``{layer: [batch tensors]}`` feature maps."""
        try:
            maps = [torch.cat(layer_outputs[layer], 0) for layer in self.layers_idxs]
        except RuntimeError as error:
            raise RuntimeError(
                "PaDiM could not stack the extracted feature maps: the clips do not all "
                "have the same duration. PaDiM fits one Gaussian per time-frequency patch, "
                "so crop or pad the clips to a fixed duration."
            ) from error
        return self._project(fuse_feature_maps(maps))

    # ----------------------------------------------------------------- fit
    def fit(self, train_dataloader, progress: bool = True) -> None:
        """Fit the per-patch Gaussians in one streaming pass, one batch in memory at a time."""
        accumulator = GaussianAccumulator(full_covariance=not self.diag_cov)
        batches = tqdm(train_dataloader, f"| PaDiM fit | {self.class_name} |") if progress else train_dataloader
        for batch in batches:
            accumulator.update(self.embed(self.batch_input(batch)))
        self._store_gaussian(accumulator)

    def fit_multivariate_gaussian(self, embedding_vectors, update_params):
        """Fit the Gaussians on an in-memory ``(B, C, H, W)`` embedding tensor."""
        accumulator = GaussianAccumulator(full_covariance=not self.diag_cov)
        accumulator.update(embedding_vectors)
        if update_params:
            self._store_gaussian(accumulator)
            return self.gauss_mean, self.gauss_cov
        return accumulator.finalize(self.covariance_reg)

    def _store_gaussian(self, accumulator: GaussianAccumulator) -> None:
        mean, covariance = accumulator.finalize(self.covariance_reg)
        self.fit_grid = accumulator.grid
        self.gauss_mean, self.gauss_cov = mean, covariance

    def train_epoch(self, epoch, train_dataloader, training_args: TrainingArgs):
        self.fit(train_dataloader)
        return 0.0

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):
        raise NotImplementedError("Audio PaDiM is fitted with fit/train_epoch, not train_step")

    # ------------------------------------------------------------- inference
    def _fitted_statistics(self, device: torch.device):
        """Means and (pre-inverted) covariances as tensors on ``device``, built once."""
        if self.gauss_mean is None or self.gauss_cov is None:
            raise RuntimeError("The model must be trained before computing distances.")
        if device not in self._inference_cache:
            mean = torch.from_numpy(np.ascontiguousarray(self.gauss_mean)).to(device)
            covariance = torch.from_numpy(np.ascontiguousarray(self.gauss_cov)).to(device)
            if covariance.ndim == 2:
                precision = 1.0 / covariance.clamp_min(1e-12)  # (C, P)
            else:
                # (C, C, P) -> (P, C, C), inverted once in float64 instead of on every batch
                precision = torch.linalg.inv(covariance.permute(2, 0, 1).double()).float()
            self._inference_cache[device] = (mean, precision)
        return self._inference_cache[device]

    def compute_distances(self, embedding_vectors: torch.Tensor) -> torch.Tensor:
        """Per-patch Mahalanobis distance, ``(B, H0, W0)`` on the embedding device.

        If the test clips have another duration than the training ones the embedding
        is resized to the training grid (with a warning) instead of crashing, because
        PaDiM stores one Gaussian per patch.
        """
        mean, precision = self._fitted_statistics(embedding_vectors.device)
        batch, channels, height, width = embedding_vectors.shape
        fitted_patches = mean.shape[1]
        fit_grid = tuple(self.fit_grid) if self.fit_grid is not None else None
        if fit_grid is not None and fit_grid[0] * fit_grid[1] != fitted_patches:
            raise ValueError(
                f"Inconsistent PaDiM state: grid {fit_grid} does not match {fitted_patches} fitted patches"
            )
        grid_matches = (height, width) == fit_grid if fit_grid is not None else height * width == fitted_patches
        if not grid_matches:
            if fit_grid is None:
                raise ValueError(
                    f"PaDiM was fitted on {fitted_patches} time-frequency patches but received "
                    f"{height * width} ({height}x{width}); the fitted grid is unknown."
                )
            if not self._warned_grid_mismatch:
                warnings.warn(
                    f"PaDiM was fitted on a {tuple(self.fit_grid)} grid but received {height}x{width}: "
                    "resizing the embedding to the training grid. Use clips of the training "
                    "duration for exact scores.",
                    stacklevel=2,
                )
                self._warned_grid_mismatch = True
            embedding_vectors = F.interpolate(
                embedding_vectors, size=tuple(self.fit_grid), mode="bilinear", align_corners=False
            )
            height, width = self.fit_grid

        delta = embedding_vectors.reshape(batch, channels, -1).float() - mean.unsqueeze(0)
        if precision.ndim == 2:
            squared = (delta * delta * precision.unsqueeze(0)).sum(dim=1)
        else:
            transformed = torch.einsum("bcp,pcd->bpd", delta, precision)
            squared = (transformed * delta.transpose(1, 2)).sum(dim=2)
        return squared.clamp_min(0).sqrt().reshape(batch, height, width)

    def forward(self, x):
        if self.training:
            # raw feature maps, kept for the legacy two-step fit (extract -> embed -> fit)
            return {layer: [maps.cpu()] for layer, maps in zip(self.layers_idxs, self._extract(x))}

        distances = self.compute_distances(self.embed(x))
        output_size = self.resolve_output_size(x, distances)
        score_map = F.interpolate(
            distances.unsqueeze(1), size=output_size, mode="bilinear", align_corners=False
        )
        anomaly_maps = gaussian_smooth(score_map, sigma=4)
        anomaly_scores = anomaly_maps.flatten(start_dim=1).amax(dim=1)
        return anomaly_maps, anomaly_scores, temporal_topk_scores(anomaly_maps)

    def resolve_output_size(self, x, dist_list):
        """Spatial size the anomaly map must be reported at."""
        if self.img_size is not None:
            return tuple(int(value) for value in self.img_size)
        if x.ndim == 4:
            # caller already passed a spectrogram / feature map
            return tuple(int(value) for value in x.shape[-2:])
        # waveform input without a declared size: keep the embedding grid
        return tuple(int(value) for value in dist_list.shape[-2:])

    # ----------------------------------------------------------- persistence
    def get_model_savepath(self, save_path):
        return os.path.join(
            save_path,
            "checkpoints_%s" % self.backbone_model_name,
            "train_%s.pth.tar" % self.class_name,
        )

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        for p in self.HYPERPARAMS:
            state_dict[p] = getattr(self, p)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        for p in self.HYPERPARAMS:
            if p in state_dict:
                setattr(self, p, state_dict[p])
        self.load_backbone(self.backbone_model)
        state_dict = {k: v for k, v in state_dict.items() if k not in self.HYPERPARAMS}
        return super().load_state_dict(state_dict, strict=strict)
