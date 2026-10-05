import pytest
import torch

from moviad.models.audio.patchcore.coreset import (
    StreamingCoreset,
    k_center_greedy,
    make_generator,
)


def test_k_center_greedy_picks_one_point_per_far_apart_cluster():
    generator = torch.Generator().manual_seed(1)
    centers = torch.tensor([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    points = torch.cat([c + 0.1 * torch.randn(50, 2, generator=generator) for c in centers])

    selected = k_center_greedy(points, 3, make_generator(0))

    assert sorted(int(i) // 50 for i in selected) == [0, 1, 2]


def test_k_center_greedy_never_repeats_and_clamps_to_the_number_of_rows():
    points = torch.randn(40, 8)
    # duplicated rows must not make the greedy loop return the same index again
    points[20:] = points[:20]

    selected = k_center_greedy(points, 1000, make_generator(0))
    assert sorted(selected.tolist()) == list(range(40))

    selected = k_center_greedy(points, 25, make_generator(0))
    assert len(set(selected.tolist())) == 25
    assert k_center_greedy(points, 0).numel() == 0


def test_k_center_greedy_is_reproducible_for_a_seed():
    points = torch.randn(300, 16)
    first = k_center_greedy(points, 20, make_generator(7))
    second = k_center_greedy(points, 20, make_generator(7))
    assert torch.equal(first, second)


def test_streaming_coreset_bounds_the_pool_and_the_bank():
    coreset = StreamingCoreset(bank_size=300, projection_dim=8, seed=0)
    for _ in range(20):
        coreset.add(torch.randn(1000, 32), keep=50)
        assert coreset.pooled <= 20 * 50

    bank = coreset.result()

    assert bank.shape == (300, 32)
    # the bank holds real embeddings, not projected ones
    assert bank.dtype == torch.float32


def test_streaming_coreset_keeps_everything_when_the_data_is_smaller_than_the_bank():
    coreset = StreamingCoreset(bank_size=1000, seed=0)
    coreset.add(torch.randn(30, 6), keep=30)
    coreset.add(torch.randn(20, 6), keep=20)

    bank = coreset.result()

    assert bank.shape == (50, 6)
    assert len({tuple(row.tolist()) for row in bank}) == 50


def test_streaming_coreset_reports_an_empty_dataset():
    with pytest.raises(RuntimeError, match="empty dataset"):
        StreamingCoreset(bank_size=10).result()
