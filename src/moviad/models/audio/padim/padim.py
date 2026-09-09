import os
from random import sample
from typing import Mapping, Union, Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from moviad.models.audio.audio_vad_model import AudioVADModel
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
        ("1", "2", "3"): (1792, 225),
        ("0", "1", "2"): (896, 225),
    },
}


def idx_to_layer_name(backbone_model_name, idx: Union[Tuple, List]):
    if backbone_model_name in ["wide_resnet50_2"]:
        return tuple(f"layer{i}" for i in idx)
    elif backbone_model_name == "mobilenet_v2":
        return tuple(f"features.{i}" for i in idx)
    else:
        return idx


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
            backbone_model_name: one of the following strings: 'wide_resnet50_2', 'mobilenet_v2'
            save_path: path to save the model and the extracted features
            class_name: one of the following strings: 'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
            diag_cov: if True, keep only the diagonal elements of the covariance matrices
        """
        super().__init__(
            feature_extractor=backbone_model,
            input_size=img_size,
            device=device,
        )
        self.class_name = class_name
        self.diag_cov = diag_cov
        self.covariance_reg = float(covariance_reg)
        # feature extractor backbone model
        self.backbone_model_name = backbone_model_name
        self.layers_idxs = layers_idxs
        self.backbone_model = self.load_backbone(backbone_model)
        self.feature_extractor = self.backbone_model
        # dimensionality reduction: random projection
        _, max_embedding_dim = self.lookup_embedding_size(backbone_model_name, layers_idxs)
        self.d = int(max_embedding_dim if embedding_dim is None else embedding_dim)
        if not 1 <= self.d <= self.t_d:
            raise ValueError(
                f"embedding_dim must be between 1 and {self.t_d}, got {self.d}"
            )
        random_dims = torch.tensor(sample(range(0, self.t_d), self.d), dtype=torch.long)
        self.random_dimensions = torch.nn.Parameter(random_dims, requires_grad=False)
        # training: learn the multivariate Gaussian distribution from the extracted features
        self.train_outputs = None  # list of mean and covariance matrix numpy arrays
        self.gauss_mean = None
        self.gauss_cov = None
        self.img_size = img_size

    def train_epoch(
        self,
        epoch,
        train_dataloader,
        training_args: TrainingArgs,
    ):
        self.train()
        layer_outputs: dict[str, list[torch.Tensor]] = {
            layer: [] for layer in self.layers_idxs
        }

        for batch in tqdm(train_dataloader, "| feature extraction | train | audio padim |"):
            outputs = self(self.batch_input(batch).to(self.device))
            assert isinstance(outputs, dict)
            for layer, output in outputs.items():
                layer_outputs[layer].extend(output)

        embedding_vectors = self.raw_feature_maps_to_embeddings(layer_outputs)
        self.fit_multivariate_gaussian(embedding_vectors, update_params=True)
        return 0.0

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):
        raise NotImplementedError("Audio PaDiM is fitted with train_epoch, not train_step")

    @staticmethod
    def embedding_concat(x, y):
        """Concatenate two feature maps, replicating ``y`` over ``x``'s grid.

        The previous unfold/fold implementation assumed ``H1`` to be an exact
        multiple of ``H2``. Audio feature maps do not satisfy that: the time
        axis follows the clip length, so ratios such as 69/34 occur regularly
        and left the trailing time frames of the output filled with zeros.
        Nearest-neighbour upsampling replicates every low resolution patch over
        its block exactly like the fold did, and is bit-identical to the old
        code whenever the ratio is an exact integer, while also covering the
        remainder rows.
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Batch size mismatch between feature maps: {x.shape[0]} and {y.shape[0]}"
            )
        y = F.interpolate(y, size=x.shape[-2:], mode="nearest")
        return torch.cat((x, y), dim=1)

    def raw_feature_maps_to_embeddings(
        self, layer_outputs: Dict[str, List[torch.Tensor]]
    ):
        """
        Given a dict of lists of outputs of the layers, concatenate the feature maps and
        eventually reduce the dimensionality to return the embedding vectors.

        - embedding vector shape: (B, C, H, W)
        - B = number of samples in the train set
        - C = number of "channels", or number of feature maps --> may be reduced by dim. reduction
        - H, W = height and width of the feature maps
        """
        # concatenate the outputs of the different dataloader batches
        try:
            output_tensors: dict[str, torch.Tensor] = {
                layer: torch.cat(outputs, 0) for layer, outputs in layer_outputs.items()
            }
        except RuntimeError as error:
            raise RuntimeError(
                "PaDiM could not stack the extracted feature maps. This happens "
                "when the training clips do not all have the same duration: PaDiM "
                "fits one Gaussian per time-frequency patch, so every clip must "
                "produce a feature map of the same size. Crop or pad the clips to "
                "a fixed duration before training."
            ) from error
        # concatenate the feature maps to get the raw embedding vectors
        embedding_vectors: torch.Tensor = output_tensors[self.layers_idxs[0]]
        for layer in self.layers_idxs[1:]:
            embedding_vectors = Padim.embedding_concat(
                embedding_vectors, output_tensors[layer]
            )
        # dimensionality reduction: select the random dimensions to reduce the embedding vectors
        assert embedding_vectors.size(1) == self.t_d, f"wront embedding size {self.t_d}, true one is: {embedding_vectors.size(1)}"
        random_dimensions = self.random_dimensions.to(embedding_vectors.device)
        embedding_vectors = torch.index_select(
            embedding_vectors, 1, random_dimensions
        )
        return embedding_vectors

    def forward(self, x):
        # 1. extract feature maps and get the raw layer outputs (conv. feature maps)
        layer_outputs: dict[str, list[torch.Tensor]] = {
            layer: [] for layer in self.layers_idxs
        }
        # forward through the net to get the intermediate outputs with the hooks
        with torch.no_grad():
            # _ = self.backbone(x)
            _ = self.backbone(x)
        # get intermediate layer outputs
        for layer, output in zip(self.layers_idxs, self.outputs):  # new
            layer_outputs[layer].append(output.cpu().detach())  # new
        # initialize hook outputs
        self.outputs = []

        if self.training:
            return layer_outputs

        # ---- EVAL INFERENCE ----
        # 2. use the feature maps to get the embeddings
        embedding_vectors = self.raw_feature_maps_to_embeddings(layer_outputs)
        # 3. compute the distance matrix
        dist_list = self.compute_distances(embedding_vectors)
        # 4. upsample to the spectrogram resolution.  ``x`` is a waveform of
        # shape [B, samples] on the benchmark loaders, so its spatial size
        # cannot be read off dimension 2 the way the visual PaDiM does.
        output_size = self.resolve_output_size(x, dist_list)
        score_map = F.interpolate(
            dist_list.unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).cpu().numpy()
        if score_map.ndim == 2:
            score_map = score_map[None, ...]
        # 5. apply gaussian smoothing on the score map.  Smooth the batch in
        # one call to avoid a Python loop over every test sample.
        score_map = gaussian_filter(score_map, sigma=(0, 4, 4))

        # 6. follow the audio contract shared with PatchCore, CFA, STFPM and
        # the spectrogram adapters: torch tensors, and a per time frame score
        # obtained by averaging the top-k frequency responses.
        anomaly_maps = torch.from_numpy(score_map).unsqueeze(1)
        anomaly_scores = anomaly_maps.flatten(start_dim=1).amax(dim=1)
        map_without_channel = anomaly_maps.squeeze(1)
        top_k = min(5, map_without_channel.shape[2])
        tmp_scores = map_without_channel.topk(top_k, dim=2).values.mean(dim=2)
        return anomaly_maps, anomaly_scores, tmp_scores

    def resolve_output_size(self, x, dist_list):
        """Spatial size the anomaly map must be reported at."""
        if self.img_size is not None:
            return tuple(int(value) for value in self.img_size)
        if x.ndim == 4:
            # caller already passed a spectrogram / feature map
            return tuple(int(value) for value in x.shape[-2:])
        # waveform input without a declared size: keep the embedding grid
        return tuple(int(value) for value in dist_list.shape[-2:])

    def fit_multivariate_gaussian(self, embedding_vectors, update_params):
        """
        Fit a multivariate Gaussian distribution to the set of given embedding vectors.

        Returns:
            List of mean and covariance matrix numpy arrays
        """
        B, C, H, W = embedding_vectors.size()

        embedding_vectors = embedding_vectors.view(B, C, H * W).float().cpu()
        mean_t = embedding_vectors.mean(dim=0)
        mean = mean_t.numpy()

        if self.diag_cov:
            # A diagonal Gaussian avoids H*W dense matrix inversions.  Keep
            # only C variances per patch instead of C*C covariance entries.
            cov = embedding_vectors.var(dim=0, unbiased=B > 1).numpy()
            cov += self.covariance_reg
        else:
            centered = embedding_vectors - mean_t.unsqueeze(0)
            denominator = max(B - 1, 1)
            cov = torch.einsum("bcp,bdp->cdp", centered, centered).numpy()
            cov /= denominator
            cov += self.covariance_reg * np.eye(C, dtype=cov.dtype)[:, :, None]
        if update_params:
            self.gauss_mean, self.gauss_cov = mean, cov
        return mean, cov

    def load_backbone(self, backbone_model):
        """
        Load the backbone model

        Args:
            backbone_model_name: one of the following strings: 'wide_resnet50_2', 'mobilenet_v2'
        """
        if backbone_model is None:
            backbone_model = AudioFeatureExtractor(
                model_name=self.backbone_model_name,
                layers_idx=self.layers_idxs,
                device=self.device,
                frozen=True,
            )

        # define the backbone behavior
        def backbone_forward(x):
            self.outputs = backbone_model(x)

        self.backbone = backbone_forward

        # save the true and random projection dimensions
        self.t_d, default_d = self.lookup_embedding_size(
            self.backbone_model_name, self.layers_idxs
        )
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

    def get_model_savepath(self, save_path):
        return os.path.join(
            save_path,
            "checkpoints_%s" % self.backbone_model_name,
            "train_%s.pth.tar" % self.class_name,
        )

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # add all the hyperparameters to the state dict
        for p in self.HYPERPARAMS:
            state_dict[p] = getattr(self, p)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # load the hyperparameters
        for p in self.HYPERPARAMS:
            if p in state_dict:
                setattr(self, p, state_dict[p])
        # load the backbone models
        self.load_backbone(self.backbone_model)
        # remove the hyperparameters from the state dict
        state_dict = {k: v for k, v in state_dict.items() if k not in self.HYPERPARAMS}
        return super().load_state_dict(state_dict, strict=strict)

    def compute_distances(self, embedding_vectors: torch.Tensor):
        """Compute Mahalanobis distances for all patches in one vectorized pass."""
        batch_size, channels, height, width = embedding_vectors.size()
        patch_count = height * width
        embeddings = embedding_vectors.view(batch_size, channels, patch_count).cpu().numpy()
        embeddings = np.moveaxis(embeddings, 1, 2)  # (batch, patch, channel)

        if self.gauss_mean is None or self.gauss_cov is None:
            raise RuntimeError("The model must be trained before computing distances.")

        fitted_patches = self.gauss_mean.shape[1]
        if patch_count != fitted_patches:
            raise ValueError(
                f"PaDiM was fitted on {fitted_patches} time-frequency patches but "
                f"received {patch_count} ({height}x{width}). The test clips must "
                "have the same duration as the training clips, because PaDiM "
                "stores one Gaussian per patch."
            )

        means = np.moveaxis(self.gauss_mean, 1, 0)  # (patch, channel)
        deltas = embeddings - means[None, :, :]
        if self.diag_cov or self.gauss_cov.ndim == 2:
            variances = np.moveaxis(self.gauss_cov, 1, 0)
            squared_distances = np.sum(
                (deltas * deltas) / np.maximum(variances[None, :, :], 1e-12),
                axis=2,
            )
        else:
            covariances = np.moveaxis(self.gauss_cov, 2, 0)  # (patch, channel, channel)
            covariance_inverses = np.linalg.inv(covariances)
            squared_distances = np.einsum(
                "bpc,pcd,bpd->bp", deltas, covariance_inverses, deltas
            )
        distances = np.sqrt(np.maximum(squared_distances, 0.0))
        return torch.from_numpy(distances.reshape(batch_size, height, width))

    def reset_model(self):
        self.train_outputs = None
        self.gauss_mean = None
        self.gauss_cov = None
