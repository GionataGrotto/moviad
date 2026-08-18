from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset, Subset


def training_subset(dataset: Dataset, fraction: float = 1.0, seed: int = 0) -> Dataset:
    """Return a deterministic random fraction of a training dataset."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"subset must be greater than 0 and at most 1, got {fraction}")
    if fraction == 1.0:
        return dataset

    dataset_size = len(dataset)
    if dataset_size == 0:
        return Subset(dataset, [])

    subset_size = max(1, math.floor(dataset_size * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)
