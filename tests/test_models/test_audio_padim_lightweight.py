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
    score_map, scores = model(inputs)
    assert score_map.shape == (6, 1, 8, 8)
    assert scores.shape == (6,)


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
    _, scores = model(inputs)
    assert scores.shape == (6,)
