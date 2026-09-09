import torch
from torch import nn

from moviad.models.audio.padim.padim import Padim


class FakeAudioFeatures(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return [
            torch.randn(batch_size, 128, 8, 8),
            torch.randn(batch_size, 256, 4, 4),
            torch.randn(batch_size, 512, 2, 2),
        ]


def test_padim_diagonal_covariance_is_compact_and_scores_batch():
    model = Padim(
        "Cnn14",
        "ToyCar",
        torch.device("cpu"),
        ["conv_block2", "conv_block3", "conv_block4"],
        diag_cov=True,
        img_size=(8, 8),
        backbone_model=FakeAudioFeatures(),
        embedding_dim=32,
    )
    inputs = torch.randn(6, 1, 8, 8)

    model.train()
    features = model(inputs)
    embeddings = model.raw_feature_maps_to_embeddings(features)
    model.fit_multivariate_gaussian(embeddings, update_params=True)

    assert model.gauss_cov.ndim == 2
    assert model.gauss_cov.shape == (32, 64)

    model.eval()
    # Audio models return (anomaly_map, image_score, temporal_score).
    score_map, scores, tmp_scores = model(inputs)
    assert score_map.shape == (6, 1, 8, 8)
    assert scores.shape == (6,)
    assert tmp_scores.shape == (6, 8)


def test_padim_full_covariance_remains_available():
    model = Padim(
        "Cnn14",
        "ToyCar",
        torch.device("cpu"),
        ["conv_block2", "conv_block3", "conv_block4"],
        diag_cov=False,
        img_size=(8, 8),
        backbone_model=FakeAudioFeatures(),
        embedding_dim=4,
    )
    inputs = torch.randn(6, 1, 8, 8)

    model.train()
    features = model(inputs)
    embeddings = model.raw_feature_maps_to_embeddings(features)
    model.fit_multivariate_gaussian(embeddings, update_params=True)

    assert model.gauss_cov.shape == (4, 4, 64)
    model.eval()
    _, scores, _ = model(inputs)
    assert scores.shape == (6,)


def test_embedding_concat_keeps_every_time_frame_on_non_integer_ratios():
    """Audio feature maps often have a non-integer time ratio (e.g. 69/34).

    The old unfold/fold implementation left the trailing time frames of the
    concatenated embedding filled with zeros, which silently corrupted the
    anomaly score at the end of every clip.
    """
    coarse_shapes = [(1, 512, 34, 4), (1, 512, 8, 4), (1, 256, 34, 8)]
    fine_shapes = [(1, 384, 138, 16), (1, 384, 34, 16), (1, 128, 69, 16)]

    for fine_shape, coarse_shape in zip(fine_shapes, coarse_shapes):
        fine = torch.ones(*fine_shape)
        coarse = torch.ones(*coarse_shape)

        concatenated = Padim.embedding_concat(fine, coarse)

        assert concatenated.shape[2:] == fine.shape[2:]
        assert concatenated.shape[1] == fine_shape[1] + coarse_shape[1]
        # no time frame may be left empty
        per_frame = concatenated.abs().sum(dim=(0, 1, 3))
        assert int((per_frame == 0).sum()) == 0


def test_padim_scores_waveforms_without_a_declared_image_size():
    """``img_size=None`` used to index dimension 2 of a 2D waveform batch."""
    model = Padim(
        "Cnn14",
        "ToyCar",
        torch.device("cpu"),
        ["conv_block2", "conv_block3", "conv_block4"],
        diag_cov=True,
        img_size=None,
        backbone_model=FakeAudioFeatures(),
        embedding_dim=32,
    )
    waveforms = torch.randn(4, 16000)

    model.train()
    features = model(waveforms)
    embeddings = model.raw_feature_maps_to_embeddings(features)
    model.fit_multivariate_gaussian(embeddings, update_params=True)

    model.eval()
    anomaly_maps, anomaly_scores, tmp_scores = model(waveforms)

    assert anomaly_maps.ndim == 4 and anomaly_maps.shape[:2] == (4, 1)
    assert anomaly_scores.shape == (4,)
    assert tmp_scores.shape == (4, anomaly_maps.shape[2])


def test_padim_reports_unsupported_layer_combinations():
    import pytest

    with pytest.raises(KeyError, match="Unsupported layer combination"):
        Padim(
            "HTSAT-base",
            "ToyCar",
            torch.device("cpu"),
            ["conv_block2", "conv_block3", "conv_block4"],
            backbone_model=FakeAudioFeatures(),
        )
