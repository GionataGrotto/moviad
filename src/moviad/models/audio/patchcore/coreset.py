"""Memory-bounded k-center-greedy coreset selection for the audio PatchCore.

Audio patch embeddings are huge (a 10 s clip yields ~2000 patches of ~900 dims,
so a thousand training clips are ~7 GB). The previous implementation stacked every
embedding in RAM, ran a sparse random projection over the whole matrix and then
looped 30000 greedy iterations over millions of rows. Here the stream is reduced
batch by batch, so peak memory is ``O(pool + batch)`` whatever the dataset size.
"""

from __future__ import annotations

import math

import torch
from tqdm import tqdm


def make_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


class RandomProjector:
    """Gaussian random projection, only used to make the greedy distances cheap."""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        self.in_dim = in_dim
        self.out_dim = min(out_dim, in_dim)
        self.matrix = None
        if self.out_dim < in_dim:
            generator = make_generator(seed)
            self.matrix = torch.randn(in_dim, self.out_dim, generator=generator) / math.sqrt(self.out_dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.matrix is None:
            return x.float()
        return x.float() @ self.matrix.to(x.device)


@torch.no_grad()
def k_center_greedy(
    features: torch.Tensor,
    n_select: int,
    generator: torch.Generator | None = None,
    desc: str | None = None,
) -> torch.Tensor:
    """Indices of ``n_select`` rows that greedily minimise the maximum distance to a center.

    Runs on ``features.device``. ``n_select`` is clamped to the number of rows and
    already selected rows are never picked twice.
    """
    n = features.shape[0]
    n_select = min(int(n_select), n)
    if n_select <= 0:
        return torch.empty(0, dtype=torch.long)
    if n_select == n:
        return torch.arange(n)

    z = features.float()
    sq_norm = (z * z).sum(dim=1)
    index = torch.randint(n, (1,), generator=generator).to(z.device)[0]
    min_dist = torch.full((n,), float("inf"), device=z.device)
    selected = torch.empty(n_select, dtype=torch.long, device=z.device)

    steps = range(n_select)
    if desc is not None and n_select >= 2000:
        steps = tqdm(steps, desc=desc, leave=False)
    for step in steps:
        selected[step] = index
        distance = sq_norm - 2.0 * (z @ z[index]) + sq_norm[index]
        min_dist = torch.minimum(min_dist, distance)
        min_dist[index] = -1.0
        index = torch.argmax(min_dist)
    return selected.cpu()


class StreamingCoreset:
    """Approximate PatchCore coreset over an arbitrarily long stream of embeddings.

    Every incoming batch is reduced to ``keep`` representatives with k-center
    greedy; at the end the pooled representatives are reduced again to
    ``bank_size``. A coreset of coresets keeps the coverage guarantee of the
    greedy selection while bounding memory.
    """

    def __init__(self, bank_size: int, projection_dim: int = 128, seed: int = 0):
        self.bank_size = int(bank_size)
        self.projection_dim = projection_dim
        self.seed = seed
        self.generator = make_generator(seed)
        self.projector: RandomProjector | None = None
        self._embeddings: list[torch.Tensor] = []
        self._projections: list[torch.Tensor] = []

    @property
    def pooled(self) -> int:
        return sum(chunk.shape[0] for chunk in self._embeddings)

    def add(self, embeddings: torch.Tensor, keep: int) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected (patches, channels) embeddings, got {tuple(embeddings.shape)}")
        if self.projector is None:
            self.projector = RandomProjector(embeddings.shape[1], self.projection_dim, self.seed)
        projected = self.projector(embeddings)
        indices = k_center_greedy(projected, keep, self.generator)
        indices_on_device = indices.to(embeddings.device)
        self._embeddings.append(embeddings[indices_on_device].detach().cpu())
        self._projections.append(projected[indices.to(projected.device)].cpu())

    def result(self, device: torch.device | str = "cpu") -> torch.Tensor:
        if not self._embeddings:
            raise RuntimeError("Cannot build a memory bank from an empty dataset")
        embeddings = torch.cat(self._embeddings, dim=0)
        if embeddings.shape[0] > self.bank_size:
            projections = torch.cat(self._projections, dim=0).to(device)
            indices = k_center_greedy(projections, self.bank_size, self.generator, desc="Coreset selection")
            embeddings = embeddings[indices]
        return embeddings
