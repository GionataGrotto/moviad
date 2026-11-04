import pytest
import torch
from torch.utils.data import DataLoader

from moviad.datasets.audio_augment import AugmentedDataset, augment_waveform, domain_sample_weights
from moviad.models.audio.patchcore.patchcore import PatchCore
from moviad.models.audio.patchcore.target_domain import fit_domain_transfer, reduce_to

BANDS, FRAMES, CHANNELS = 4, 8, 6


class LevelExtractor:
    """Waveform [B, FRAMES * BANDS] -> feature map whose level is the waveform itself, so that a domain is an offset."""

    quantized = False

    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, device):
        return self

    def __call__(self, batch):
        grid = batch.reshape(batch.shape[0], 1, FRAMES, BANDS)
        return [grid.expand(-1, CHANNELS, -1, -1).contiguous()]


def make_model(**kwargs):
    defaults = dict(device=torch.device("cpu"), input_size=(FRAMES, BANDS), feature_extractor=LevelExtractor(), memory_bank_size=64,
                    num_neighbors=3, blur=False)
    defaults.update(kwargs)
    model = PatchCore(**defaults)
    model.feature_pooler = torch.nn.Identity()
    return model


def clips(n, offset, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(torch.randn(n, FRAMES * BANDS, generator=generator) * 0.1 + offset, batch_size=4)


@pytest.mark.parametrize("band_banks", [False, True])
def test_target_quota_puts_target_patches_in_the_bank(band_banks):
    plain = make_model(band_banks=band_banks)
    plain.fit(clips(40, 0.0, 0))
    assert plain.memory_bank.mean().abs() < 0.2  # only source patches

    model = make_model(band_banks=band_banks, target_bank_fraction=0.25)
    model.fit(clips(40, 0.0, 0), target_dataloader=clips(4, 5.0, 1))
    assert model.memory_bank.shape[0] == 64
    near_target = (model.memory_bank.mean(dim=1) > 2.5).sum().item()
    assert near_target == 16  # a quarter of the bank, reserved for the target patches


def test_pseudo_target_patches_follow_the_target_statistics():
    model = make_model(pseudo_target_fraction=0.5, domain_transfer="meanstd")
    model.fit(clips(40, 0.0, 0), target_dataloader=clips(4, 5.0, 1))
    assert model.memory_bank.shape[0] == 64
    moved = (model.memory_bank.mean(dim=1) > 2.5).sum().item()
    assert moved == 32  # half of the bank: source patches carried to the target level


def test_target_options_need_the_target_clips():
    model = make_model(target_bank_fraction=0.25)
    with pytest.raises(ValueError, match="target_dataloader"):
        model.fit(clips(8, 0.0, 0))
    with pytest.raises(ValueError):
        make_model(target_bank_fraction=0.6, pseudo_target_fraction=0.5)


def test_reduce_to_and_the_two_domain_transfers():
    x = torch.randn(200, 5)
    assert reduce_to(x, 50).shape == (50, 5) and reduce_to(x, 500).shape == (200, 5) and reduce_to(x, 0).shape == (0, 5)
    source = torch.randn(2000, 5)
    target = torch.randn(2000, 5) * torch.tensor([1.0, 2.0, 0.5, 3.0, 1.0]) + 4.0
    for mode in ("meanstd", "coral"):
        moved = fit_domain_transfer(source, target, mode, shrinkage=0.0)(source)
        assert torch.allclose(moved.mean(dim=0), target.mean(dim=0), atol=0.15), mode
        assert torch.allclose(moved.std(dim=0), target.std(dim=0), rtol=0.15), mode
    with pytest.raises(ValueError):
        fit_domain_transfer(source, target, "other")


def test_augmentation_is_deterministic_creates_new_clips_and_keeps_the_length():
    base = [torch.randn(16000) for _ in range(3)]
    augmented = AugmentedDataset(base, copies=4, seed=7, mix_with=[torch.randn(16000)])
    assert len(augmented) == 12
    first, again = augmented[5], augmented[5]
    assert torch.equal(first, again) and first.shape == (16000,)
    assert not torch.allclose(augmented[1], base[1])
    assert not torch.allclose(augmented[1], augmented[4])  # two copies of the same clip differ
    generator = torch.Generator().manual_seed(0)
    assert augment_waveform(torch.zeros(16000), generator).shape == (16000,)


def test_domain_weights():
    domains = ["source"] * 8 + ["target"] * 2
    assert domain_sample_weights(domains, 3.0) == [1.0] * 8 + [3.0] * 2
    weights = domain_sample_weights(domains, "balance")
    assert sum(weights[:8]) == pytest.approx(sum(weights[8:]))


def test_speed_change_uses_a_small_integer_ratio(monkeypatch):
    """A random sample rate (16000 -> 15521) makes torchaudio build a ~1 GB kernel per call: the ratio has to be k / 100."""
    import torchaudio

    calls = []
    real = torchaudio.functional.resample

    def spy(wave, orig, new, *args, **kwargs):
        calls.append((orig, new))
        return real(wave, orig, new, *args, **kwargs)

    monkeypatch.setattr(torchaudio.functional, "resample", spy)
    generator = torch.Generator().manual_seed(0)
    for _ in range(40):
        augment_waveform(torch.randn(16000), generator, probability=1.0)
    assert calls and all(orig == 100 and 97 <= new <= 103 for orig, new in calls)


@pytest.mark.parametrize("band_banks", [False, True])
def test_a_quota_the_target_clips_cannot_fill_goes_back_to_the_source(band_banks):
    """One target clip gives few patches: the bank keeps its size, the unused part of the quota holds source patches."""
    model = make_model(band_banks=band_banks, target_bank_fraction=0.75)
    model.fit(clips(40, 0.0, 0), target_dataloader=clips(1, 5.0, 1))
    assert model.memory_bank.shape[0] == 64
    target_patches = FRAMES * BANDS  # one clip
    assert (model.memory_bank.mean(dim=1) > 2.5).sum().item() == target_patches  # all of them, and no more than exist
