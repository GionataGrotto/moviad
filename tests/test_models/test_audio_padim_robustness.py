import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from moviad.models.audio.padim.padim import GaussianAccumulator, Padim

LAYERS = ["conv_block2", "conv_block3", "conv_block4"]


class DeterministicFeatures(nn.Module):
    """Spectrogram [B, 1, H, W] -> three maps with 128/256/512 channels, a pure function of x."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.freq = torch.rand(896, generator=generator) * 3
        self.phase = torch.rand(896, generator=generator) * 6

    def forward(self, x):
        fine = torch.sin(x * self.freq[:128].view(1, -1, 1, 1) + self.phase[:128].view(1, -1, 1, 1))
        mid = torch.sin(F.avg_pool2d(x, 2) * self.freq[128:384].view(1, -1, 1, 1) + self.phase[128:384].view(1, -1, 1, 1))
        coarse = torch.sin(F.avg_pool2d(x, 4) * self.freq[384:].view(1, -1, 1, 1) + self.phase[384:].view(1, -1, 1, 1))
        return [fine, mid, coarse]


def make_model(diag_cov, **kwargs):
    defaults = dict(
        backbone_model_name="Cnn14",
        class_name="ToyCar",
        device=torch.device("cpu"),
        layers_idxs=LAYERS,
        diag_cov=diag_cov,
        img_size=(8, 8),
        backbone_model=DeterministicFeatures(),
        embedding_dim=6,
        covariance_reg=0.01,
    )
    defaults.update(kwargs)
    return Padim(**defaults)


def spectrogram_loader(num_clips=10, batch_size=3, shape=(8, 8), seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.randn(num_clips, 1, *shape, generator=generator)), batch_size=batch_size)


def reference_statistics(embeddings, reg, diag):
    """Textbook per-patch mean / covariance of a stacked (N, C, H, W) tensor."""
    n, c, h, w = embeddings.shape
    x = embeddings.double().reshape(n, c, h * w).numpy()
    mean = x.mean(axis=0)
    covariance = np.stack([np.cov(x[:, :, p], rowvar=False).reshape(c, c) for p in range(h * w)], axis=-1)
    covariance = covariance + reg * np.eye(c)[:, :, None]
    return mean, (np.einsum("ccp->cp", covariance) if diag else covariance)


@pytest.mark.parametrize("diag", [True, False])
def test_streaming_accumulator_matches_the_statistics_of_the_stacked_dataset(diag):
    embeddings = torch.randn(11, 5, 4, 3) * 3 + 2
    accumulator = GaussianAccumulator(full_covariance=not diag)
    for chunk in embeddings.split([1, 4, 2, 4]):  # uneven batches, including a singleton
        accumulator.update(chunk)

    mean, covariance = accumulator.finalize(covariance_reg=0.01)
    expected_mean, expected_covariance = reference_statistics(embeddings, 0.01, diag)

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(covariance, expected_covariance, rtol=1e-4, atol=1e-5)


def test_accumulator_rejects_a_grid_that_changes_mid_fit():
    accumulator = GaussianAccumulator(full_covariance=False)
    accumulator.update(torch.randn(2, 3, 4, 4))
    with pytest.raises(ValueError, match="same duration"):
        accumulator.update(torch.randn(2, 3, 5, 4))


@pytest.mark.parametrize("diag", [True, False])
def test_streaming_fit_equals_fitting_the_stacked_embeddings(diag):
    streamed = make_model(diag)
    streamed.fit(spectrogram_loader(), progress=False)

    stacked = make_model(diag)
    stacked.random_dimensions = streamed.random_dimensions
    spectrograms = torch.cat([batch[0] for batch in spectrogram_loader()])
    stacked.fit_multivariate_gaussian(stacked.embed(spectrograms), update_params=True)

    np.testing.assert_allclose(streamed.gauss_mean, stacked.gauss_mean, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(streamed.gauss_cov, stacked.gauss_cov, rtol=1e-4, atol=1e-5)
    assert streamed.fit_grid == (8, 8)


def naive_mahalanobis(model, spectrograms):
    """The original per-call numpy implementation (inverts every covariance on every call)."""
    embeddings = model.embed(spectrograms).numpy()
    b, c, h, w = embeddings.shape
    embeddings = np.moveaxis(embeddings.reshape(b, c, h * w), 1, 2)
    deltas = embeddings - np.moveaxis(model.gauss_mean, 1, 0)[None]
    if model.gauss_cov.ndim == 2:
        squared = np.sum(deltas**2 / np.moveaxis(model.gauss_cov, 1, 0)[None], axis=2)
    else:
        inverses = np.linalg.inv(np.moveaxis(model.gauss_cov, 2, 0))
        squared = np.einsum("bpc,pcd,bpd->bp", deltas, inverses, deltas)
    return np.sqrt(np.maximum(squared, 0)).reshape(b, h, w)


@pytest.mark.parametrize("diag", [True, False])
def test_vectorised_distances_match_the_naive_reference(diag):
    model = make_model(diag)
    model.fit(spectrogram_loader(), progress=False)
    test_clips = torch.randn(4, 1, 8, 8)

    distances = model.compute_distances(model.embed(test_clips))

    np.testing.assert_allclose(distances.numpy(), naive_mahalanobis(model, test_clips), rtol=2e-3, atol=2e-3)


def test_covariance_inverse_is_computed_once_not_on_every_batch(monkeypatch):
    model = make_model(diag_cov=False)
    model.fit(spectrogram_loader(), progress=False)
    model.eval()
    calls = []
    original = torch.linalg.inv
    monkeypatch.setattr(torch.linalg, "inv", lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs))

    for _ in range(3):
        model(torch.randn(2, 1, 8, 8))

    assert len(calls) == 1

    model.fit(spectrogram_loader(seed=1), progress=False)  # refitting must invalidate the cache
    model(torch.randn(2, 1, 8, 8))
    assert len(calls) == 2


def test_clips_with_another_duration_are_resized_with_a_warning_instead_of_crashing():
    model = make_model(diag_cov=True, img_size=None)
    model.fit(spectrogram_loader(shape=(8, 8)), progress=False)
    model.eval()
    longer = torch.randn(2, 1, 12, 8)  # 12 time frames instead of 8

    with pytest.warns(UserWarning, match="resizing the embedding"):
        maps, scores, temporal = model(longer)

    assert maps.shape == (2, 1, 12, 8)  # reported at the input size
    assert torch.isfinite(scores).all() and temporal.shape == (2, 12)


def test_matching_clips_do_not_warn():
    model = make_model(diag_cov=True)
    model.fit(spectrogram_loader(), progress=False)
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model(torch.randn(2, 1, 8, 8))


def test_a_backbone_returning_the_wrong_number_of_maps_is_reported_clearly():
    class TwoMaps(nn.Module):
        def forward(self, x):
            return [torch.randn(x.shape[0], 128, 8, 8), torch.randn(x.shape[0], 256, 4, 4)]

    model = make_model(diag_cov=True, backbone_model=TwoMaps())
    with pytest.raises(ValueError, match="returned 2 feature maps"):
        model.fit(spectrogram_loader(), progress=False)


def test_fit_accepts_tuple_batches_and_waveform_loaders():
    model = make_model(diag_cov=True, img_size=None)

    class WaveformBackbone(nn.Module):
        def forward(self, x):
            return DeterministicFeatures()(x.reshape(x.shape[0], 1, 8, 8))

    model = make_model(diag_cov=True, img_size=None, backbone_model=WaveformBackbone())
    dataset = TensorDataset(torch.randn(6, 64), torch.zeros(6))  # (waveform, label) batches
    model.fit(DataLoader(dataset, batch_size=2), progress=False)
    model.eval()

    maps, scores, _ = model(torch.randn(2, 64))

    assert maps.shape[:2] == (2, 1) and scores.shape == (2,)


def test_state_dict_round_trip_restores_the_fitted_gaussians():
    fitted = make_model(diag_cov=False)
    fitted.fit(spectrogram_loader(), progress=False)
    fitted.eval()

    restored = make_model(diag_cov=False)
    restored.load_state_dict(fitted.state_dict())
    restored.eval()
    clips = torch.randn(2, 1, 8, 8)

    assert restored.fit_grid == fitted.fit_grid
    for expected, actual in zip(fitted(clips), restored(clips)):
        torch.testing.assert_close(actual, expected)


def test_reset_model_forgets_the_fit():
    model = make_model(diag_cov=True)
    model.fit(spectrogram_loader(), progress=False)
    model.reset_model()
    model.eval()
    with pytest.raises(RuntimeError, match="must be trained"):
        model(torch.randn(1, 1, 8, 8))
