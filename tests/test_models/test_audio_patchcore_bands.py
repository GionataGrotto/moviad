import pytest
import torch
from torch.utils.data import DataLoader

from moviad.models.audio.components.band_scores import standardize_by_band
from moviad.models.audio.patchcore.patchcore import PatchCore

BANDS = 4
FRAMES = 8
CHANNELS = 6
OFFSET = 10.0


class BandExtractor:
    """Waveform [B, FRAMES * BANDS] -> one feature map whose level depends on the band: band w sits at w * OFFSET."""

    quantized = False

    def __init__(self, noise=(0.1,) * BANDS):
        self.device = torch.device("cpu")
        self.noise = torch.tensor(noise).view(1, 1, 1, BANDS)

    def to(self, device):
        return self

    def __call__(self, batch):
        grid = batch.reshape(batch.shape[0], 1, FRAMES, BANDS)
        offsets = torch.arange(BANDS, dtype=torch.float32).view(1, 1, 1, BANDS) * OFFSET
        return [(grid * self.noise + offsets).expand(-1, CHANNELS, -1, -1).contiguous()]


def make_model(**kwargs):
    defaults = dict(
        device=torch.device("cpu"),
        input_size=(FRAMES, BANDS),
        feature_extractor=BandExtractor(),
        memory_bank_size=64,
        num_neighbors=3,
        blur=False,
    )
    defaults.update(kwargs)
    model = PatchCore(**defaults)
    model.feature_pooler = torch.nn.Identity()  # keep the bands apart: the 3x3 average would mix neighbouring ones
    return model


def loader(clips=40, batch_size=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(torch.randn(clips, FRAMES * BANDS, generator=generator), batch_size=batch_size)


def test_one_bank_per_band_with_the_memory_split_evenly():
    model = make_model(band_banks=True)
    model.fit(loader())

    assert model.band_bank_sizes.tolist() == [16] * BANDS  # 64 patches split over 4 bands
    assert model.memory_bank.shape == (64, CHANNELS)
    # every bank holds only patches of its own band: band w lives around w * OFFSET
    offsets = model._band_offsets()
    for band in range(BANDS):
        bank = model.memory_bank[offsets[band] : offsets[band + 1]]
        assert abs(float(bank.mean()) - band * OFFSET) < 0.5


def test_a_patch_is_only_compared_with_normal_patches_of_its_own_band():
    """A band-0-like patch at band 3 is normal for a global bank but anomalous for per-band banks."""
    global_model, banded = make_model(), make_model(band_banks=True)
    global_model.fit(loader())
    banded.fit(loader())
    global_model.eval(), banded.eval()

    clip = torch.randn(1, FRAMES * BANDS) * 0.1
    clip = clip.reshape(1, FRAMES, BANDS)
    clip[:, 3, 3] = -(3 * OFFSET) / 0.1  # feature value 0 where band 3 should sit at 30
    clip = clip.reshape(1, -1)

    _, global_score, _ = global_model(clip)
    _, banded_score, _ = banded(clip)

    assert float(banded_score) > 10 * float(global_score)


def test_per_band_banks_also_work_with_the_default_neighbour_reweighting():
    model = make_model(band_banks=True, num_neighbors=9)
    model.fit(loader())
    model.eval()

    maps, scores, temporal = model(torch.randn(3, FRAMES * BANDS))

    assert maps.shape == (3, 1, FRAMES, BANDS) and torch.isfinite(scores).all() and temporal.shape == (3, FRAMES)


def test_band_banks_validate_their_inputs():
    with pytest.raises(ValueError, match="too small"):
        make_model(band_banks=True, memory_bank_size=3).fit(loader())

    model = make_model(band_banks=True)
    model.fit(loader())
    model.eval()
    wider = PatchCore(torch.device("cpu"), (FRAMES, 2), BandExtractor(), blur=False, band_banks=True)
    wider.load_state_dict(model.state_dict())
    wider.eval()
    with pytest.raises(ValueError, match="4 frequency bands"):
        wider._patch_scores(torch.randn(1, CHANNELS, FRAMES, 5))


def test_score_standardisation_uses_held_out_clips_that_are_not_in_the_bank():
    model = make_model(score_standardization="band", holdout_fraction=0.1)
    model.fit(loader(clips=40, batch_size=2))  # 20 batches -> batches 9 and 19 are held out

    assert model.score_mean.shape == (BANDS,) and model.score_std.shape == (BANDS,)
    assert model.memory_bank.shape[0] <= 64
    held_out_patches = 2 * 2 * FRAMES  # 2 batches x 2 clips x 8 frames, per band
    assert held_out_patches == 32 and bool((model.score_std > 0).all())


def test_standardised_scores_of_fresh_normal_clips_are_on_the_same_scale_in_every_band():
    # band 0 is much noisier than band 3: its raw patch scores are far larger
    noisy = BandExtractor(noise=(1.0, 0.3, 0.1, 0.05))
    raw = make_model(feature_extractor=noisy)
    raw.fit(loader(clips=200))
    standardised = make_model(feature_extractor=noisy, score_standardization="band")
    standardised.fit(loader(clips=200))

    fresh = noisy(torch.randn(100, FRAMES * BANDS))[0]
    raw_scores, _ = raw._patch_scores(fresh)
    standardised_scores = standardize_by_band(
        standardised._patch_scores(fresh)[0], standardised.score_mean, standardised.score_std
    )

    raw_per_band = raw_scores.mean(dim=(0, 1))
    assert float(raw_per_band.max() / raw_per_band.min()) > 2.5  # the noisy band dominates a plain max
    per_band_mean = standardised_scores.mean(dim=(0, 1))
    per_band_std = standardised_scores.std(dim=(0, 1))
    assert float(per_band_mean.abs().max()) < 0.5 and float((per_band_std - 1).abs().max()) < 0.5


def test_score_standardisation_needs_held_out_batches():
    model = make_model(score_standardization="band")
    with pytest.raises(RuntimeError, match="held-out"):
        model.fit(loader(clips=4, batch_size=2))  # 2 batches: none is the 10th


def test_state_dict_round_trip_keeps_banks_and_score_statistics():
    fitted = make_model(band_banks=True, score_standardization="band")
    fitted.fit(loader())
    fitted.eval()

    restored = make_model(band_banks=True, score_standardization="band")
    restored.load_state_dict(fitted.state_dict())
    restored.eval()
    clips = torch.randn(2, FRAMES * BANDS)

    assert torch.equal(restored.band_bank_sizes, fitted.band_bank_sizes)
    for expected, actual in zip(fitted(clips), restored(clips)):
        torch.testing.assert_close(actual, expected)


def test_checkpoints_without_the_band_buffers_still_load():
    plain = make_model()
    plain.fit(loader())
    legacy = {k: v for k, v in plain.state_dict().items() if k not in ("band_bank_sizes", "score_mean", "score_std")}

    restored = make_model()
    restored.load_state_dict(legacy)  # strict: the new buffers are optional

    assert torch.equal(restored.memory_bank, plain.memory_bank)


@pytest.mark.parametrize("band_banks", [False, True])
def test_patch_reweighting_scales_scores_into_zero_and_the_plain_score(band_banks):
    plain = make_model(band_banks=band_banks)
    plain.fit(loader())
    weighted = make_model(band_banks=band_banks, patch_reweighting=True)
    weighted.fit(loader())
    plain.eval(), weighted.eval()
    clips = torch.randn(3, FRAMES * BANDS, generator=torch.Generator().manual_seed(5)) + 0.5
    plain_map, _, _ = plain(clips)
    weighted_map, _, _ = weighted(clips)
    assert weighted_map.shape == plain_map.shape
    assert (weighted_map <= plain_map + 1e-5).all()  # the weight is in (0, 1) and the scores are non-negative
    assert (weighted_map < plain_map).any()
