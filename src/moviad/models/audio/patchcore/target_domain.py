"""Memory-bank helpers for the few target-domain clips: a reserved quota and source patches moved to the target statistics.

With 990 source clips and 10 target ones a k-center memory bank keeps only what is far from the rest, and duplicating
the target clips does not change it (a duplicate is never the farthest point). Two things that do:

* a quota of the bank reserved for the target patches (all of them, or a k-center reduction when there are more);
* pseudo-target patches: source patches mapped to the mean / covariance of the target ones. The map has to go from
  source to target, not the other way round: the domain of a test clip is unknown, so a test patch is never aligned
  and has to find its match in a bank that contains the target statistics.
"""

from __future__ import annotations

import torch

from moviad.models.audio.patchcore.coreset import RandomProjector, k_center_greedy, make_generator


def reduce_to(embeddings: torch.Tensor, size: int, seed: int = 0, projection_dim: int = 128) -> torch.Tensor:
    """At most ``size`` rows of ``embeddings``, chosen by k-center greedy (all of them if there are fewer)."""
    if size <= 0:
        return embeddings[:0]
    if embeddings.shape[0] <= size:
        return embeddings
    projected = RandomProjector(embeddings.shape[1], projection_dim, seed)(embeddings)
    indices = k_center_greedy(projected, size, make_generator(seed))
    return embeddings[indices.to(embeddings.device)]


def _sqrt_and_inverse_sqrt(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh(covariance)
    values = values.clamp_min(1e-8)
    return (vectors * values.sqrt()) @ vectors.T, (vectors / values.sqrt()) @ vectors.T


def _covariance(x: torch.Tensor, shrinkage: float) -> torch.Tensor:
    centred = x - x.mean(dim=0, keepdim=True)
    covariance = centred.T @ centred / max(x.shape[0] - 1, 1)
    identity = torch.eye(x.shape[1], dtype=x.dtype, device=x.device) * covariance.diagonal().mean().clamp_min(1e-12)
    return (1 - shrinkage) * covariance + shrinkage * identity


def fit_domain_transfer(source: torch.Tensor, target: torch.Tensor, mode: str = "meanstd", shrinkage: float = 0.5):
    """A function that maps source patches ``(N, C)`` to the statistics of the target patches.

    ``meanstd``: per-channel mean and standard deviation. ``coral``: mean and (shrunk) covariance, ``x' = (x - mu_s) Cs^-1/2 Ct^1/2 + mu_t``;
    the shrinkage towards a scaled identity matters because the covariance has thousands of channels and the target only a few thousand patches.
    """
    source, target = source.double(), target.double()
    mean_s, mean_t = source.mean(dim=0), target.mean(dim=0)
    if mode == "meanstd":
        scale = target.std(dim=0).clamp_min(1e-6) / source.std(dim=0).clamp_min(1e-6)
        return lambda x: ((x.double() - mean_s) * scale + mean_t).to(x.dtype)
    if mode == "coral":
        _, inverse_sqrt_s = _sqrt_and_inverse_sqrt(_covariance(source, shrinkage))
        sqrt_t, _ = _sqrt_and_inverse_sqrt(_covariance(target, shrinkage))
        transform = inverse_sqrt_s @ sqrt_t
        return lambda x: (((x.double() - mean_s) @ transform) + mean_t).to(x.dtype)
    raise ValueError(f"domain_transfer must be 'meanstd' or 'coral', got {mode!r}")
