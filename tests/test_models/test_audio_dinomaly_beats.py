from types import SimpleNamespace

import pytest
import torch
from torch import nn

from moviad.models.audio.dinomaly_beats import BEATsDinomaly
from moviad.models.dinomaly.dinomaly import DinomalyTrainArgs
from moviad.trainers.audio.trainer_dinomaly import TrainerDinomaly

DIM, SIDE = 16, 16


class ToyExtractor(nn.Module):
    """Waveform [B, 256] -> two (B, 16, 4, 4) maps, like the hooked blocks of BEATs; frozen."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.model = SimpleNamespace()
        self.convs = nn.ModuleList([nn.Conv2d(1, DIM, kernel_size=4, stride=4) for _ in range(2)])
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.spec_shape = None

    def forward(self, waveforms):
        grid = waveforms.reshape(waveforms.shape[0], 1, SIDE, SIDE)
        self.spec_shape = torch.Size([waveforms.shape[0], 1, 20, 24])  # the map is reported at the filterbank resolution
        return [conv(grid) for conv in self.convs]


def normal_clips(n=64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(1, SIDE * SIDE, generator=generator)
    return base + 0.05 * torch.randn(n, SIDE * SIDE, generator=generator)


def make_model(**kwargs):
    return BEATsDinomaly(ToyExtractor(), torch.device("cpu"), dim=DIM, num_heads=2, decoder_blocks=2, **kwargs)


def test_the_frozen_encoder_is_not_trained_and_the_contract_holds():
    model = make_model()
    model.train()
    encoder, decoder = model(normal_clips(4))
    assert len(encoder) == len(decoder) == 1 and encoder[0].shape == decoder[0].shape == (4, DIM, 4, 4)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.trainable.parameters())
    model.eval()
    maps, scores, temporal = model(normal_clips(4))
    assert maps.shape == (4, 1, 20, 24) and scores.shape == (4,) and temporal.shape == (4, 20)
    assert torch.allclose(scores, maps.flatten(1).amax(dim=1))


@pytest.mark.parametrize("distance", ["cosine", "l2"])
def test_training_lowers_the_loss_and_separates_a_different_signal(distance):
    model = make_model(distance=distance)
    loader = torch.utils.data.DataLoader(normal_clips(), batch_size=8, shuffle=True)
    losses = TrainerDinomaly(model, loader, "cpu").train(epochs=12, batch_size=8)
    assert losses[-1] < losses[0]
    model.eval()
    with torch.no_grad():
        normal = model(normal_clips(8, seed=1))[1].mean()
        anomalous = model(torch.randn(8, SIDE * SIDE, generator=torch.Generator().manual_seed(5)))[1].mean()
    assert anomalous > normal


def test_unknown_distance_and_missing_size_are_rejected():
    with pytest.raises(ValueError, match="distance"):
        make_model(distance="manhattan")
    with pytest.raises(ValueError, match="embedding size"):
        BEATsDinomaly(ToyExtractor(), torch.device("cpu"))


def test_train_args_come_from_the_visual_implementation():
    args = DinomalyTrainArgs(batch_size=8, epochs=1)
    args.total_iters = 10
    model = make_model()
    model.train()
    args.init_train(model)
    assert model.train_step(normal_clips(8), args) > 0
