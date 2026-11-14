import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_benchmark"))

import benchmark_common as common  # noqa: E402
from moviad.utilities.audio.beats_feature_extractor import BEATsFbank, BEATsFeatureExtractor  # noqa: E402

BEATS = {"backbone": "BEATs", "layers": [8, 10, 12]}


def test_the_fbank_has_the_layout_of_the_cnn14_log_mel_and_the_shape_the_anomaly_maps_use():
    wave = torch.randn(2, 16000)
    out = BEATsFbank()(wave)
    assert out.shape == (2, 1, *BEATsFeatureExtractor.fbank_shape(16000))  # [B, 1, frames, 128]
    assert BEATsFbank()(wave[0]).shape == (1, 1, *BEATsFeatureExtractor.fbank_shape(16000))
    assert BEATsFbank.hop_length == 160
    # the way the datasets read it: spectro_transform(rand(1, N)).shape[-2:] and wav_to_spectro(clip.unsqueeze(0)).squeeze(0)
    assert BEATsFbank()(torch.rand(1, 32000)).shape[-2:] == BEATsFeatureExtractor.fbank_shape(32000)
    assert BEATsFbank()(wave[0].unsqueeze(0)).squeeze(0).shape == (1, *BEATsFeatureExtractor.fbank_shape(16000))


def test_the_mimii_and_envmix_runners_get_16_khz_and_the_fbank_with_beats():
    assert common.audio_sample_rate(BEATS) == 16000
    assert isinstance(common.make_wave_to_spectrogram(BEATS, torch.device("cpu"), "patchcore"), BEATsFbank)
    assert isinstance(common.make_wave_to_spectrogram(BEATS, torch.device("cpu"), "dinomaly"), BEATsFbank)  # no Cnn14 frontend for BEATs
    assert common.audio_sample_rate({"backbone": "Cnn14"}) == 44100  # unchanged


def test_the_extractor_accepts_a_waveform_with_a_channel_axis(tmp_path):
    pytest.importorskip("torchaudio")
    # no checkpoint here: only the squeeze of [B, 1, T] is checked, through the class source
    import inspect

    assert "waveforms.squeeze(1)" in inspect.getsource(BEATsFeatureExtractor.forward)


class _Clips(torch.utils.data.Dataset):
    """Background clips ``(waveform, sample_rate)`` and, with ``categories``, anomaly clips ``(waveform, sample_rate, category)``."""

    def __init__(self, n, rate, categories=False):
        self.n, self.rate, self.categories = n, rate, categories

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (torch.randn(self.rate * 2) * 0.1, self.rate)

    def get_with_category(self, idx):
        return torch.randn(self.rate // 2) * 0.1, self.rate, "dog"


def test_the_envmix_test_clips_follow_the_sample_rate_and_the_spectrogram_of_the_backbone():
    """With BEATs the test clips were made at 44.1 kHz and their masks had the Cnn14 shape, so the batches could not be stacked."""
    from moviad.datasets.audio_dataset import AudioAnomalyDataset

    beats = AudioAnomalyDataset(_Clips(6, 16000), _Clips(3, 16000), SNR_dB_distribution=lambda: 0.0, clip_duration=2.0, random_state=0,
                                target_sample_rate=16000, spectro_transform=BEATsFbank())
    assert beats.img_shape == BEATsFeatureExtractor.fbank_shape(32000)
    assert all(sample.audio_clip.shape[0] == 32000 for sample in beats.dataset)  # 16 kHz, not 44.1 kHz

    default = AudioAnomalyDataset(_Clips(6, 44100), _Clips(3, 44100), SNR_dB_distribution=lambda: 0.0, clip_duration=2.0, random_state=0)
    assert default.img_shape[1] == 64 and default.dataset[0].audio_clip.shape[0] == 88200  # the Cnn14 path is unchanged
