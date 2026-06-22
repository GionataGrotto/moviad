"""k-Center Greedy Method.

Returns points that minimizes the maximum distance of any point to a center.
- https://arxiv.org/abs/1708.00489
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from sklearn.random_projection import SparseRandomProjection
import numpy as np
from tqdm import tqdm

class KCenterGreedy:
    """Implements k-center-greedy method.

    Args:
        embedding (torch.Tensor): Embedding vector extracted from a CNN
        sampling_ratio (float): Ratio to choose coreset size from the embedding size.

    Example:
        >>> embedding.shape
        torch.Size([219520, 1536])
        >>> sampler = KCenterGreedy(embedding=embedding)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        torch.Size([219, 1536])
    """

    def __init__(self, embeddings: torch.Tensor, device: torch.device, sampling_ratio: float = 0.1) -> None:

        self.embeddings = embeddings
        self.coreset_size = int(embeddings.shape[0] * sampling_ratio)
        self.projector = SparseRandomProjection(n_components="auto", eps=0.90)

        self.features: torch.Tensor
        self.min_distances: torch.Tensor = None
        self.n_observations = self.embeddings.shape[0]

        self.device = device


    def reset_distances(self):
        self.min_distances = None

    def update_distances(self, cluster_centers:list):
        """Update min distances given cluster centers.

        Args:
            cluster_centers (list[int]): indices of cluster centers
        """

        if cluster_centers:
            centers = self.features[cluster_centers]

            distances = F.pairwise_distance(self.features, centers, p = 2).reshape(-1,1)

            if self.min_distances is None:
                self.min_distances = distances
            else:
                self.min_distances = torch.minimum(self.min_distances, distances)

    def get_new_idx(self) -> int:
        """Get index value of a sample.

        Based on minimum distance of the cluster

        Returns:
            int: Sample index
        """

        if isinstance(self.min_distances, torch.Tensor):
            idx = int(torch.argmax(self.min_distances).item())
        else:
            msg = f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}"
            raise TypeError(msg)

        return idx

    def get_coreset_idx_randomp(
        self,
        z_lib,
        n : int = 1000,
        eps : float = 0.90,
        float16 : bool = True,
        force_cpu : bool = False,
        memory_bank_size = 30000
    ):
        """Returns n coreset idx for given z_lib.

        Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
        CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)

        Args:
            z_lib:      (n, d) tensor of patches.
            n:          Number of patches to select.
            eps:        Agression of the sparse random projection.
            float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
            force_cpu:  Force cpu, useful in case of GPU OOM.

        Returns:
            coreset indices
        """

        print(z_lib.device)
        print(z_lib.shape)

        print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
        try:
            transformer = SparseRandomProjection(eps=eps)

            z_lib = torch.tensor(transformer.fit_transform(z_lib))
            print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
        except ValueError:
            print( "   Error: could not project vectors. Please increase `eps`.")

        select_idx = 0
        last_item = z_lib[select_idx:select_idx+1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True)
        # The line below is not faster than linalg.norm, although i'm keeping it in for
        # future reference.
        # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

        if float16:
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        if torch.cuda.is_available() and not force_cpu:
            last_item = last_item.to("cuda")
            z_lib = z_lib.to("cuda")
            min_distances = min_distances.to("cuda")

        
        for _ in tqdm(range(memory_bank_size), desc="Compute coreset distance"):
            distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True) # broadcasting step
            # distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True) # broadcasting step
            min_distances = torch.minimum(distances, min_distances) # iterative step
            select_idx = torch.argmax(min_distances) # selection step

            # bookkeeping
            last_item = z_lib[select_idx:select_idx+1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))

        return torch.stack(coreset_idx)

    # def select_coreset_idxs(self, selected_idxs: list[int] | None = None) -> list[int]:
    #     """Greedily form a coreset to minimize the maximum distance of a cluster.

    #     Args:
    #         selected_idxs: index of samples already selected. Defaults to an empty set.

    #     Returns:
    #       indices of samples selected to minimize distance to cluster centers
    #     """

    #     if selected_idxs is None:
    #         selected_idxs = []

    #     print(f"Fitting random projections. Start dim = {self.embeddings.shape}.")
    #     if self.embeddings.ndim == 2:
    #         embedding_np = self.embeddings.cpu().numpy()
    #         self.projector.fit(embedding_np)
    #         self.features = torch.from_numpy(self.projector.transform(embedding_np)).to(self.device)
    #         self.reset_distances()
    #     else:
    #         self.features = self.embedding.reshape(self.embedding.shape[0], -1)
    #         self.update_distances(cluster_centers=selected_idxs)

    #     print(f"DONE: Final dim = {self.features.shape}")

    #     selected_coreset_idx: list[int] = []
    #     idx = int(torch.randint(high = self.n_observations, size =(1,)).item())

    #     for _ in tqdm(range(30000)):
    #         self.update_distances(cluster_centers = [idx])
    #         idx = self.get_new_idx()

    #         if idx in selected_idxs:
    #             msg = "New indices should not be in selected indices."
    #             raise ValueError(msg)

    #         self.min_distances[idx] = 0
    #         selected_coreset_idx.append(idx)

    #     return selected_coreset_idx