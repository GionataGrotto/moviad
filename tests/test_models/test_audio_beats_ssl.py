import pytest
import torch
from torch import nn

from moviad.models.audio.beats_ssl import (
    adapt_beats, info_nce, load_adapter, lora_off, masked_input, save_adapter, token_loss, tokens,
)

FRAMES, BINS, PATCH, DIM = 32, 16, 8, 12


class ToyBEATs(nn.Module):
    """A minimal stand-in with the interface the adaptation uses: preprocess, patch_embedding, extract_features, q_proj / v_proj."""

    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(1, DIM, kernel_size=PATCH, stride=PATCH)
        self.blocks = nn.ModuleList([nn.Module(), nn.Module()])
        for block in self.blocks:
            block.q_proj, block.v_proj = nn.Linear(DIM, DIM), nn.Linear(DIM, DIM)

    def preprocess(self, source, **kwargs):
        return source.reshape(source.shape[0], FRAMES, BINS)

    def extract_features(self, source, padding_mask=None, **kwargs):
        x = self.patch_embedding(self.preprocess(source).unsqueeze(1)).flatten(2).transpose(1, 2).transpose(0, 1)  # (N, B, D)
        for block in self.blocks:
            x = x + block.v_proj(torch.tanh(block.q_proj(x)))
            for hook in block._forward_hooks.values():
                hook(block, None, x)
        return x


class ToyExtractor:
    layers = (1, 2)

    def __init__(self):
        torch.manual_seed(0)
        self.model = ToyBEATs()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self._captured = {}
        for layer, block in zip(self.layers, self.model.blocks):
            block.register_forward_hook(lambda _m, _i, out, layer=layer: self._captured.__setitem__(layer, out))


class Clips(torch.utils.data.Dataset):
    def __init__(self, n=12):
        generator = torch.Generator().manual_seed(1)
        near = torch.randn(n, FRAMES * BINS, generator=generator)
        self.clips = torch.stack([near, near + 0.3 * torch.randn(n, FRAMES * BINS, generator=generator)], dim=1)  # (n, 2, T)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return self.clips[i]


@pytest.mark.parametrize("objective", ["reg", "mask", "contrastive"])
def test_every_objective_trains_the_adapters_and_keeps_the_base_weights(objective):
    extractor = ToyExtractor()
    base = {k: v.clone() for k, v in extractor.model.state_dict().items()}
    history = adapt_beats(extractor, Clips(), objective=objective, epochs=3, batch_size=4, lr=1e-2, rank=2, seed=0, log=lambda *_: None)
    assert len(history) == 3 and all(torch.isfinite(torch.tensor(history)))
    state = extractor.model.state_dict()
    assert any(k.endswith("lora_b.weight") and state[k].abs().sum() > 0 for k in state)  # the adapters moved
    for name, tensor in base.items():  # the original weights did not (they live under .base now)
        key = name.replace("q_proj.", "q_proj.base.").replace("v_proj.", "v_proj.base.")
        assert torch.equal(state[key], tensor), name
    assert not any(p.requires_grad for p in extractor.model.parameters())


def test_the_frozen_network_is_the_one_with_the_adapters_off():
    extractor = ToyExtractor()
    wave = Clips()[0][0].unsqueeze(0)
    before = [t.clone() for t in tokens(extractor, wave)]
    adapt_beats(extractor, Clips(), objective="reg", epochs=2, batch_size=4, lr=1e-2, rank=2, log=lambda *_: None)
    with torch.no_grad():
        adapted = tokens(extractor, wave)
        with lora_off(extractor.model):
            frozen = tokens(extractor, wave)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(frozen, before))
    assert any(not torch.allclose(a, b, atol=1e-6) for a, b in zip(adapted, before))


def test_masked_input_hides_patches_and_restores_the_model():
    extractor = ToyExtractor()
    wave = torch.randn(3, FRAMES * BINS) + 1.0
    plain = extractor.model.preprocess(wave)
    with masked_input(extractor.model, 0.5, torch.Generator().manual_seed(0), (PATCH, PATCH)) as state:
        hidden = extractor.model.preprocess(wave)
    assert state["mask"].shape == (3, FRAMES // PATCH, BINS // PATCH) and state["mask"].any()
    assert (hidden == 0).any() and not (plain == 0).any()
    assert torch.equal(extractor.model.preprocess(wave), plain)  # the original method is back


def test_losses_and_adapter_roundtrip(tmp_path):
    a = torch.randn(2, 5, DIM)
    assert token_loss([a], [a]).item() == pytest.approx(0.0, abs=1e-6)
    assert token_loss([a], [-a]).item() == pytest.approx(2.0, abs=1e-5)
    z = torch.randn(6, DIM)
    assert info_nce(z, z, 0.1) < info_nce(z, z.roll(1, dims=0), 0.1)

    extractor = ToyExtractor()
    adapt_beats(extractor, Clips(), objective="contrastive", epochs=2, batch_size=4, lr=1e-2, rank=2, log=lambda *_: None)
    wave = Clips()[3][0].unsqueeze(0)
    with torch.no_grad():
        expected = tokens(extractor, wave)
    save_adapter(extractor, tmp_path / "adapter.pt", rank=2, alpha=16.0)
    fresh = ToyExtractor()
    load_adapter(fresh, tmp_path / "adapter.pt")
    with torch.no_grad():
        loaded = tokens(fresh, wave)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(expected, loaded))


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match="objective"):
        adapt_beats(ToyExtractor(), Clips(), objective="other")
    with pytest.raises(ValueError, match="signal"):
        adapt_beats(ToyExtractor(), Clips(), signal="far")
