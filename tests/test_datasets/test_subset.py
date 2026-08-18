import pytest
import torch
from torch.utils.data import TensorDataset

from moviad.datasets.subset import training_subset


def test_training_subset_halves_dataset_deterministically():
    dataset = TensorDataset(torch.arange(11))

    first = training_subset(dataset, 0.5, seed=42)
    second = training_subset(dataset, 0.5, seed=42)

    assert len(first) == 5
    assert first.indices == second.indices


def test_training_subset_defaults_to_original_dataset():
    dataset = TensorDataset(torch.arange(3))

    assert training_subset(dataset) is dataset


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.01])
def test_training_subset_rejects_invalid_fraction(fraction):
    dataset = TensorDataset(torch.arange(3))

    with pytest.raises(ValueError, match="subset"):
        training_subset(dataset, fraction)
