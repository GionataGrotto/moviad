import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from moviad.models.audio.patchcore import patchcore as patchcore_module
from moviad.models.audio.patchcore.anomaly_map import AnomalyMapGenerator
from moviad.models.audio.patchcore.patchcore import PatchCore


class DeterministicExtractor:
    """Waveform [B, 256] -> two feature maps that are a pure function of the clip."""

    quantized = False

    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        return self

    def __call__(self, batch):
        spectrogram = batch.reshape(batch.shape[0], 1, 16, 16)
        channels = torch.linspace(0.5, 2.0, 6).reshape(1, 6, 1, 1)
        fine = torch.sin(spectrogram * channels + channels)
        return [fine, F.avg_pool2d(fine, 2)]


def make_model(**kwargs):
    defaults = dict(
        device=torch.device("cpu"),
        input_size=(16, 16),
        feature_extractor=DeterministicExtractor(),
        memory_bank_size=64,
        num_neighbors=3,
        blur=True,
    )
    defaults.update(kwargs)
    return PatchCore(**defaults)


def loader(num_clips=12, batch_size=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.randn(num_clips, 256, generator=generator)), batch_size=batch_size)


def test_fit_builds_a_bounded_bank_and_restores_the_training_mode():
    model = make_model(memory_bank_size=50)
    model.eval()

    model.fit(loader())

    assert model.memory_bank.shape == (50, 12)  # 6 fine + 6 pooled channels
    assert not model.training


def test_bank_is_never_larger_than_the_number_of_patches_and_has_no_duplicates():
    model = make_model(memory_bank_size=10_000)

    model.fit(loader(num_clips=2, batch_size=2))

    assert model.memory_bank.shape[0] == 2 * 16 * 16
    assert len({tuple(row.tolist()) for row in model.memory_bank}) == model.memory_bank.shape[0]


def test_fit_streams_batches_instead_of_stacking_the_dataset(monkeypatch):
    seen = []
    original_add = patchcore_module.StreamingCoreset.add

    def recording_add(self, embeddings, keep):
        seen.append((embeddings.shape[0], keep, self.pooled))
        return original_add(self, embeddings, keep)

    monkeypatch.setattr(patchcore_module.StreamingCoreset, "add", recording_add)
    model = make_model(memory_bank_size=100)

    model.fit(loader(num_clips=16, batch_size=2), pool_factor=2)

    assert len(seen) == 8
    assert all(rows == 2 * 16 * 16 for rows, _, _ in seen)
    # every batch contributes pool_factor * bank / num_batches representatives, so the
    # pool grows linearly and never reaches the size of the dataset
    assert all(keep == 25 for _, keep, _ in seen)
    assert max(pooled for _, _, pooled in seen) <= 2 * 100


def test_scores_do_not_depend_on_which_clips_share_the_batch():
    """Per-batch min-max normalisation made a clip's score depend on its batch mates."""
    model = make_model()
    model.fit(loader())
    model.eval()
    clips = torch.randn(4, 256)
    clips[3] *= 5  # an outlier that would stretch a per-batch normalisation

    maps_all, scores_all, temporal_all = model(clips)
    for index in range(4):
        maps_one, score_one, temporal_one = model(clips[index : index + 1])
        torch.testing.assert_close(maps_one[0], maps_all[index], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(score_one[0], scores_all[index], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(temporal_one[0], temporal_all[index], rtol=1e-4, atol=1e-5)


def test_batch_normalised_maps_remain_available_on_request():
    maps = AnomalyMapGenerator(blur=False, normalize=True)(torch.rand(2, 1, 4, 4) * 10, image_size=(8, 8))
    assert float(maps.min()) == 0.0 and float(maps.max()) == 1.0


def test_chunked_nearest_neighbour_search_matches_the_unchunked_one():
    model = make_model(num_neighbors=1)
    model.fit(loader())
    model.eval()
    clips = torch.randn(3, 256)

    reference = model(clips)
    model.DISTANCE_CHUNK_ELEMENTS = model.memory_bank.shape[0] * 7  # force 7-row chunks
    chunked = model(clips)

    for expected, actual in zip(reference, chunked):
        torch.testing.assert_close(actual, expected)


def test_neighbour_reweighting_works_with_a_tiny_bank():
    model = make_model(memory_bank_size=1, num_neighbors=9)
    model.fit(loader())
    model.eval()

    _, scores, _ = model(torch.randn(2, 256))

    assert scores.shape == (2,) and torch.isfinite(scores).all()


def test_state_dict_round_trip_restores_a_fitted_memory_bank():
    fitted = make_model()
    fitted.fit(loader())
    fitted.eval()

    restored = make_model()
    restored.load_state_dict(fitted.state_dict())
    restored.eval()
    clips = torch.randn(2, 256)

    assert torch.equal(restored.memory_bank, fitted.memory_bank)
    for expected, actual in zip(fitted(clips), restored(clips)):
        torch.testing.assert_close(actual, expected)


def test_evaluating_before_fitting_reports_a_clear_error():
    model = make_model()
    model.eval()
    with pytest.raises(RuntimeError, match="memory bank is empty"):
        model(torch.randn(1, 256))


def test_blur_works_on_maps_smaller_than_the_smoothing_kernel():
    model = make_model(input_size=(8, 8), blur=True)  # kernel radius 16 > 8
    model.fit(loader())
    model.eval()

    maps, scores, temporal = model(torch.randn(2, 256))

    assert maps.shape == (2, 1, 8, 8) and torch.isfinite(maps).all()
    assert scores.shape == (2,) and temporal.shape == (2, 8)
