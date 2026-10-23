import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from moviad.models.audio.components.band_scores import standardize_by_band
from moviad.models.audio.padim.padim import Padim

LAYERS = ["conv_block2", "conv_block3", "conv_block4"]
BANDS = 8


class BandFeatures(nn.Module):
    """Spectrogram [B, 1, T, 8] -> three maps (128/256/512 channels) whose level depends on the band."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.freq = torch.rand(896, generator=generator) * 3
        self.phase = torch.rand(896, generator=generator) * 6

    def forward(self, x):
        band_level = torch.arange(x.shape[-1], dtype=torch.float32).view(1, 1, 1, -1)

        def level(maps, low, high):
            return torch.sin(maps * self.freq[low:high].view(1, -1, 1, 1) + self.phase[low:high].view(1, -1, 1, 1))

        shifted = x + band_level
        return [level(shifted, 0, 128), level(F.avg_pool2d(shifted, 2), 128, 384), level(F.avg_pool2d(shifted, 4), 384, 896)]


def make_model(diag_cov, **kwargs):
    defaults = dict(
        backbone_model_name="Cnn14", class_name="fan", device=torch.device("cpu"), layers_idxs=LAYERS,
        diag_cov=diag_cov, img_size=(8, BANDS), backbone_model=BandFeatures(), embedding_dim=6,
        covariance_reg=0.01, per_band=True,
    )
    defaults.update(kwargs)
    return Padim(**defaults)


def spectrograms(count=12, batch_size=3, frames=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.randn(count, 1, frames, BANDS, generator=generator)), batch_size=batch_size)


def band_samples(model, loader):
    """Every (clip, time frame) of a band is one sample: (samples, channels, bands)."""
    embeddings = torch.cat([model.embed(batch[0]) for batch in loader])  # (N, C, T, W)
    return embeddings.permute(0, 2, 1, 3).reshape(-1, embeddings.shape[1], embeddings.shape[3]).double().numpy()


@pytest.mark.parametrize("diag", [True, False])
def test_per_band_gaussians_use_every_time_frame_as_a_sample(diag):
    model = make_model(diag)
    model.fit(spectrograms(), progress=False)

    x = band_samples(model, spectrograms())  # (96, C, W)
    expected_mean = x.mean(axis=0)
    np.testing.assert_allclose(model.gauss_mean, expected_mean, rtol=1e-4, atol=1e-5)
    covariance = np.stack([np.cov(x[:, :, band], rowvar=False) for band in range(BANDS)], axis=-1)
    covariance = covariance + 0.01 * np.eye(6)[:, :, None]
    expected = np.einsum("ccw->cw", covariance) if diag else covariance
    np.testing.assert_allclose(model.gauss_cov, expected, rtol=1e-4, atol=1e-5)
    assert model.gauss_mean.shape == (6, BANDS)  # per band, not per (time, band) patch
    assert model.fit_grid == (1, BANDS)


def test_per_patch_gaussians_remain_the_default():
    model = make_model(True, per_band=False)
    model.fit(spectrograms(), progress=False)
    assert model.gauss_mean.shape == (6, 8 * BANDS)


@pytest.mark.parametrize("diag", [True, False])
def test_band_distances_match_a_naive_mahalanobis_computation(diag):
    model = make_model(diag)
    model.fit(spectrograms(), progress=False)
    clips = torch.randn(4, 1, 8, BANDS)

    distances = model.compute_distances(model.embed(clips)).numpy()

    embedding = model.embed(clips).numpy()  # (B, C, T, W)
    expected = np.zeros(embedding.shape[:1] + embedding.shape[2:])
    for band in range(BANDS):
        covariance = np.diag(model.gauss_cov[:, band]) if diag else model.gauss_cov[:, :, band]
        inverse = np.linalg.inv(covariance)
        delta = embedding[:, :, :, band] - model.gauss_mean[:, band].reshape(1, -1, 1)
        expected[:, :, band] = np.sqrt(np.einsum("bct,cd,bdt->bt", delta, inverse, delta))
    np.testing.assert_allclose(distances, expected, rtol=2e-3, atol=2e-3)


def test_per_band_model_scores_clips_of_any_duration_without_warning():
    model = make_model(True, img_size=None)
    model.fit(spectrograms(frames=8), progress=False)
    model.eval()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        maps, scores, temporal = model(torch.randn(2, 1, 20, BANDS))  # 20 frames instead of 8

    assert maps.shape == (2, 1, 20, BANDS) and torch.isfinite(scores).all() and temporal.shape == (2, 20)


def test_per_band_model_rejects_another_number_of_bands():
    model = make_model(True)
    model.fit(spectrograms(), progress=False)
    with pytest.raises(ValueError, match="8 frequency bands"):
        model.compute_distances(torch.randn(1, 6, 8, 5))


def test_score_standardisation_puts_the_bands_of_fresh_normal_clips_on_one_scale():
    model = make_model(True, score_standardization="band", holdout_fraction=0.25)
    model.fit(spectrograms(count=80, batch_size=2), progress=False)  # 40 batches, every 4th held out

    assert model.score_mean.shape == (BANDS,) and model.score_std.shape == (BANDS,)
    fresh = model.compute_distances(model.embed(torch.randn(60, 1, 8, BANDS)))
    standardised = standardize_by_band(fresh, model.score_mean, model.score_std)

    per_band_mean = standardised.mean(dim=(0, 1))
    per_band_std = standardised.std(dim=(0, 1))
    assert float(per_band_mean.abs().max()) < 0.5 and float((per_band_std - 1).abs().max()) < 0.5


def test_score_standardisation_needs_held_out_batches():
    model = make_model(True, score_standardization="band")
    with pytest.raises(RuntimeError, match="held-out"):
        model.fit(spectrograms(count=6, batch_size=3), progress=False)


def test_state_dict_round_trip_keeps_the_band_gaussians_and_score_statistics():
    fitted = make_model(False, score_standardization="band", holdout_fraction=0.25)
    fitted.fit(spectrograms(count=80, batch_size=2), progress=False)
    fitted.eval()

    restored = make_model(False, score_standardization="band", holdout_fraction=0.25)
    restored.load_state_dict(fitted.state_dict())
    restored.eval()
    clips = torch.randn(2, 1, 8, BANDS)

    assert restored.per_band and torch.equal(restored.score_mean, fitted.score_mean)
    for expected, actual in zip(fitted(clips), restored(clips)):
        torch.testing.assert_close(actual, expected)


def test_invalid_standardisation_is_rejected():
    with pytest.raises(ValueError, match="score_standardization"):
        make_model(True, score_standardization="global")


def test_precision_matrices_invert_a_batch_and_fall_back_when_not_positive_definite():
    from moviad.models.audio.padim.padim import precision_matrices

    generator = torch.Generator().manual_seed(0)
    a = torch.randn(5, 12, 12, generator=generator, dtype=torch.double) * 0.3
    spd = a @ a.transpose(1, 2) + 0.01 * torch.eye(12, dtype=torch.double)
    identity = torch.eye(12, dtype=torch.double)

    torch.testing.assert_close(precision_matrices(spd) @ spd, identity.expand(5, -1, -1), atol=1e-9, rtol=0)

    indefinite = torch.diag(torch.tensor([2.0, -1.0, 3.0], dtype=torch.double)).unsqueeze(0)  # not positive definite
    torch.testing.assert_close(precision_matrices(indefinite), torch.linalg.inv(indefinite))
