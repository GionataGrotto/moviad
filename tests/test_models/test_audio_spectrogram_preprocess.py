import pytest
import torch
from torch import nn

from moviad.models.audio.components.spectrogram_preprocess import SpectrogramPreprocessor, first_band_above
from paper_benchmark import benchmark_common as common


def make_spectrograms(n_batches=4, batch=3, frames=40, bands=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    offsets = torch.linspace(-30.0, 10.0, bands)  # every band has its own stationary level
    spreads = torch.linspace(0.5, 4.0, bands)
    return [
        offsets + spreads * torch.randn(batch, 1, frames, bands, generator=generator) for _ in range(n_batches)
    ]


def test_first_band_above_matches_the_cnn14_frontend():
    # 64 mel bands, 50-14000 Hz: the bands above the 8 kHz Nyquist of 16 kHz audio
    start = first_band_above(8000, 64, 50, 14000)
    assert 50 <= start <= 56
    assert first_band_above(20000, 64, 50, 14000) == 64


def test_band_zscore_equalises_the_bands_but_keeps_the_overall_range():
    spectrograms = make_spectrograms()
    preprocessor = SpectrogramPreprocessor(n_mels=8, band_norm="zscore")
    preprocessor.fit(spectrograms)

    processed = torch.cat([preprocessor(s) for s in spectrograms])
    original = torch.cat(spectrograms)

    band_mean = processed.mean(dim=(0, 1, 2))
    band_std = processed.std(dim=(0, 1, 2))
    assert band_mean.std() < 1e-3 and band_std.std() < 1e-2  # all bands now look alike ...
    torch.testing.assert_close(processed.mean(), original.mean(), atol=1e-2, rtol=0)  # ... around the original range
    torch.testing.assert_close(processed.std(), original.std(), atol=2e-2, rtol=0)


def test_dead_bands_are_replaced_by_the_last_live_band_before_the_statistics_are_taken():
    spectrograms = [s.clone() for s in make_spectrograms()]
    for s in spectrograms:
        s[..., 6:] = -100.0  # the resampling floor
    preprocessor = SpectrogramPreprocessor(n_mels=8, band_norm="none", dead_band_start=6)

    processed = preprocessor(spectrograms[0])

    assert torch.equal(processed[..., 6], spectrograms[0][..., 5]) and torch.equal(processed[..., 7], spectrograms[0][..., 5])
    assert torch.equal(processed[..., :6], spectrograms[0][..., :6])  # live bands untouched

    both = SpectrogramPreprocessor(n_mels=8, band_norm="zscore", dead_band_start=6)
    both.fit(spectrograms)
    out = both(spectrograms[0])
    assert float(out.min()) > -60  # the -100 dB floor no longer reaches the network


def test_preprocessor_validates_its_arguments_and_needs_a_fit():
    with pytest.raises(ValueError, match="band_norm"):
        SpectrogramPreprocessor(band_norm="minmax")
    with pytest.raises(ValueError, match="dead_band_start"):
        SpectrogramPreprocessor(n_mels=8, dead_band_start=0)
    unfitted = SpectrogramPreprocessor(n_mels=8, band_norm="zscore")
    with pytest.raises(RuntimeError, match="fitted"):
        unfitted(torch.zeros(1, 1, 4, 8))
    with pytest.raises(RuntimeError, match="empty"):
        unfitted.fit([])
    assert not SpectrogramPreprocessor(n_mels=8).active and SpectrogramPreprocessor(n_mels=8, band_norm="zscore").active


class FakeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.received = None

    def set_spectrogram_preprocessor(self, preprocessor):
        self.received = preprocessor


class FakeSTFPM:
    teacher, student = FakeExtractor(), FakeExtractor()


@pytest.mark.parametrize("method, consumers", [
    ("patchcore", lambda m: [m.feature_extractor]),
    ("cfa", lambda m: [m.feature_extractor]),
    ("padim", lambda m: [m.backbone_model]),
    ("stfpm", lambda m: [m.teacher, m.student]),
    ("dinomaly", lambda m: [m]),
])
def test_preprocessing_is_attached_to_every_consumer_of_the_spectrogram(method, consumers):
    model = FakeExtractor()
    model.feature_extractor = FakeExtractor()
    model.backbone_model = FakeExtractor()
    model.teacher, model.student = FakeExtractor(), FakeExtractor()
    config = {"backbone": "Cnn14", "spec_dead_band_hz": 8000}  # dead bands only: no statistics pass needed

    common.attach_spectrogram_preprocessing(method, model, [], config, torch.device("cpu"))

    attached = [c.received for c in consumers(model)]
    assert all(isinstance(p, SpectrogramPreprocessor) and p.dead_band_start is not None for p in attached)
    assert len({id(p) for p in attached}) == 1  # teacher and student share one preprocessor


def test_preprocessing_is_a_no_op_by_default():
    model = FakeExtractor()
    model.feature_extractor = FakeExtractor()
    common.attach_spectrogram_preprocessing("patchcore", model, [], {"backbone": "Cnn14"}, torch.device("cpu"))
    assert model.feature_extractor.received is None
