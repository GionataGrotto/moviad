import sys
import textwrap

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from moviad.models.audio.knn import AudioKNN, BEATsEncoder
from paper_benchmark.run_dcase2026_task2 import _aggregate_dcase_score


class ToyEncoder:
    """Two 'layers': the first 4 samples of the clip, and their running mean."""

    def to(self, device):
        return self

    def __call__(self, waveforms):
        head = waveforms[:, :4]
        return [head, head.cumsum(dim=1) / torch.arange(1, 5)]


def normal_clips(n=40, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, 16, generator=generator) * 0.1 + torch.tensor([1.0, 2.0, 3.0, 4.0] + [0.0] * 12)


def fitted_model(**kwargs):
    model = AudioKNN(ToyEncoder(), torch.device("cpu"), **kwargs)
    model.fit(DataLoader(TensorDataset(normal_clips()), batch_size=8))
    model.eval()
    return model


def test_anomalies_score_higher_than_normal_clips():
    model = fitted_model()
    normal = normal_clips(10, seed=1)
    anomalous = normal.clone()
    anomalous[:, :4] += torch.tensor([3.0, -3.0, 3.0, -3.0])

    _, normal_scores, _ = model(normal)
    _, anomaly_scores, _ = model(anomalous)

    assert anomaly_scores.min() > normal_scores.max()


def test_training_clips_do_not_match_themselves():
    model = fitted_model()
    _, scores, _ = model(normal_clips())  # the very clips in the memory
    assert (scores > 0).all() and torch.isfinite(scores).all()

    keeps_self = fitted_model(ignore_exact_matches=False)
    assert float(keeps_self(normal_clips())[1].max()) < 1e-6


def test_top_k_averages_more_neighbours_and_layers_are_averaged():
    clips = normal_clips(10, seed=2)
    nearest = fitted_model(top_k=1, metric="euclidean")(clips)[1]
    averaged = fitted_model(top_k=5, metric="euclidean")(clips)[1]
    assert (averaged >= nearest - 1e-6).all()
    assert nearest.shape == (10,)


@pytest.mark.parametrize("aggregation", ["max", "mean", "temporal_topk_mean"])
def test_output_works_with_every_dcase_score_aggregation(aggregation):
    model = fitted_model()
    output = model(normal_clips(6, seed=3))
    expected = output[1].numpy()

    scores = _aggregate_dcase_score(output, aggregation, top_k=5)

    np.testing.assert_allclose(scores, expected, rtol=1e-5)


def test_errors_are_explicit():
    with pytest.raises(ValueError, match="metric"):
        AudioKNN(ToyEncoder(), "cpu", metric="manhattan")
    model = AudioKNN(ToyEncoder(), "cpu")
    model.eval()
    with pytest.raises(RuntimeError, match="memory is empty"):
        model(torch.randn(2, 16))
    with pytest.raises(RuntimeError, match="empty dataset"):
        model.fit([])


FAKE_BEATS = textwrap.dedent(
    '''
    import torch
    from torch import nn

    class BEATsConfig:
        def __init__(self, cfg):
            self.dim = cfg["dim"]
            self.layers = cfg["layers"]

    class Block(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):  # time-first [T, B, D], returns a tuple like the real blocks
            return self.linear(x), None

    class Encoder(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.layers = nn.ModuleList([Block(cfg.dim) for _ in range(cfg.layers)])

    class BEATs(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.encoder = Encoder(cfg)

        def extract_features(self, source, padding_mask=None):
            assert padding_mask is not None and padding_mask.shape == source.shape
            x = source.unfold(-1, self.cfg.dim, self.cfg.dim)  # [B, T, D]
            x = x.transpose(0, 1)  # time-first
            for layer in self.encoder.layers:
                x, _ = layer(x)
            return x.transpose(0, 1), padding_mask
    '''
)


@pytest.fixture
def fake_beats(tmp_path):
    (tmp_path / "BEATs.py").write_text(FAKE_BEATS)
    sys.modules.pop("BEATs", None)
    sys.path.insert(0, str(tmp_path))
    import BEATs  # noqa: PLC0415

    config = {"dim": 8, "layers": 4}
    reference = BEATs.BEATs(BEATs.BEATsConfig(config))
    checkpoint = tmp_path / "beats.pt"
    torch.save({"cfg": config, "model": reference.state_dict()}, checkpoint)
    yield tmp_path, checkpoint, reference
    sys.path.remove(str(tmp_path))
    sys.modules.pop("BEATs", None)


def test_beats_encoder_returns_the_time_average_of_the_requested_blocks(fake_beats):
    code_dir, checkpoint, reference = fake_beats
    encoder = BEATsEncoder(checkpoint, code_dir=code_dir, layers=(2, 4))
    waveforms = torch.randn(3, 8 * 5)

    embeddings = encoder(waveforms)

    # independent computation of the 2nd and 4th block outputs
    x = waveforms.unfold(-1, 8, 8).transpose(0, 1)
    expected = []
    for index, block in enumerate(reference.encoder.layers, start=1):
        x = block(x)[0]
        if index in (2, 4):
            expected.append(x.transpose(0, 1).mean(dim=1))
    assert len(embeddings) == 2 and embeddings[0].shape == (3, 8)
    for actual, wanted in zip(embeddings, expected):
        torch.testing.assert_close(actual, wanted)


def test_beats_encoder_validates_layers_and_paths(fake_beats, tmp_path):
    code_dir, checkpoint, _ = fake_beats
    with pytest.raises(ValueError, match="block numbers"):
        BEATsEncoder(checkpoint, code_dir=code_dir, layers=(5,))
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        BEATsEncoder(tmp_path / "missing.pt", code_dir=code_dir)


def test_beats_encoder_feeds_the_knn_detector_end_to_end(fake_beats):
    code_dir, checkpoint, _ = fake_beats
    encoder = BEATsEncoder(checkpoint, code_dir=code_dir, layers=(3,))
    model = AudioKNN(encoder, torch.device("cpu"), top_k=1)
    train = DataLoader(TensorDataset(torch.randn(12, 40)), batch_size=4)
    model.fit(train)
    model.eval()

    _, scores, _ = model(torch.randn(5, 40))

    assert scores.shape == (5,) and torch.isfinite(scores).all()
