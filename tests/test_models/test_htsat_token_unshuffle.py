"""Regression tests for the HTSAT-base token layout fix.

HTSAT squares a (time, frequency) spectrogram before feeding it to its Swin
backbone by slicing the time axis into ``freq_ratio`` chunks and stacking them
along the frequency axis (see ``HTSAT_Swin_Transformer.reshape_wav2img`` in
laion_clap). The intermediate Swin-stage token grids this project hooks are
therefore not a plain (time, frequency) map: their height mixes (chunk index,
frequency) and their width is "time within one chunk". These tests pin the
un-scrambling in ``AudioFeatureExtractor._unshuffle_htsat_tokens`` and the
windowing in ``_forward_htsat_windowed``, both independent of the laion_clap
package (they only exercise plain tensor reshapes), so they run without the
optional HTSAT dependency installed.
"""

import torch

from moviad.utilities.audio.audio_feature_extractor import AudioFeatureExtractor


def _scramble(real_map: torch.Tensor, freq_ratio: int) -> torch.Tensor:
    """Turn a genuine (B, C, freq, time) map into HTSAT's square token layout.

    This is the exact inverse of ``_unshuffle_htsat_tokens``: split the time
    axis into ``freq_ratio`` contiguous chunks and stack them along frequency,
    matching what ``reshape_wav2img`` does to the spectrogram before the Swin
    backbone ever sees it. Used here to build inputs with a known ground truth.
    """
    batch, channels, freq_sub, time = real_map.shape
    side = freq_ratio * freq_sub
    assert time == freq_ratio * side, "time must equal freq_ratio * (freq_ratio * freq_sub)"
    x = real_map.reshape(batch, channels, freq_sub, freq_ratio, side)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    return x.reshape(batch, channels, side, side)


def test_unshuffle_htsat_tokens_is_the_inverse_of_the_scramble():
    freq_ratio = 4
    torch.manual_seed(0)
    real_map = torch.randn(2, 3, 8, 128)  # (B, C, freq=8, time=128) -> side=32

    scrambled = _scramble(real_map, freq_ratio)
    assert scrambled.shape == (2, 3, 32, 32)  # square token grid

    recovered = AudioFeatureExtractor._unshuffle_htsat_tokens(scrambled, freq_ratio)

    assert recovered.shape == real_map.shape
    assert torch.allclose(recovered, real_map)


def test_unshuffle_htsat_tokens_preserves_chronological_order():
    """A marker placed late in time must land late in the unshuffled map.

    Before this fix, downstream code (F.interpolate to the spectrogram size,
    the top-k-over-frequency temporal score) consumed the raw scrambled grid
    as if its width axis were already real time. On the scrambled grid a late
    marker can land anywhere depending on which quarter-chunk it fell into.
    """
    freq_ratio = 4
    real_map = torch.zeros(1, 1, 4, 64)  # freq_sub=4 -> side=16, time=64
    real_map[0, 0, 2, 62] = 1.0  # near the end of the real timeline

    scrambled = _scramble(real_map, freq_ratio)
    recovered = AudioFeatureExtractor._unshuffle_htsat_tokens(scrambled, freq_ratio)

    peak_freq, peak_time = (recovered[0, 0] == 1.0).nonzero()[0].tolist()
    assert (peak_freq, peak_time) == (2, 62)


class _FakeHTSAT:
    """Minimal stand-in reproducing the fixed-window contract of HTSAT-base.

    ``spec_size`` / ``freq_ratio`` mirror the real model (256 / 4), so
    ``spec_size * freq_ratio == 1024`` matches the true 10.24s-at-48kHz
    window this project's frontend produces. ``forward_features`` returns a
    feature map whose time length is proportional to the padded window length
    divided by a fixed downsampling factor, exactly like a real Swin stage
    would for a full ``target_T``-frame input.
    """

    spec_size = 256
    freq_ratio = 4

    def __init__(self, downsample: int, channels: int):
        self.downsample = downsample
        self.channels = channels
        self.calls: list[int] = []

    def bn0(self, x):
        return x

    def reshape_wav2img(self, x):
        return x

    def forward_features(self, x):
        # x here stands in for the post-reshape square image; only its
        # (batch, time) extent is used to size the fake per-layer output.
        batch, _, time_len, _ = x.shape
        self.calls.append(time_len)
        out_time = time_len // self.downsample
        return torch.arange(out_time, dtype=torch.float32).reshape(1, 1, out_time).expand(
            batch, self.channels, out_time
        ).clone()


def test_forward_htsat_windowed_matches_single_window_for_a_short_clip():
    extractor = AudioFeatureExtractor.__new__(AudioFeatureExtractor)
    extractor.model = _FakeHTSAT(downsample=16, channels=4)

    def fake_single_window(spectrogram):
        extractor.model.forward_features(spectrogram)
        # emulate one layer's hooked, already-unshuffled feature map
        time_len = spectrogram.shape[2] // extractor.model.downsample
        return [torch.arange(time_len).float().reshape(1, 1, 1, time_len)]

    extractor._forward_htsat_single_window = fake_single_window

    target_T = extractor.model.spec_size * extractor.model.freq_ratio
    short = torch.zeros(1, 1, target_T // 2, 64)

    result = AudioFeatureExtractor._forward_htsat_windowed(extractor, short)

    # a clip within the window must never trigger this path in forward(); this
    # test only pins that windowed and single-window math agree when it is
    # invoked directly with a single, fully-real window.
    assert len(result) == 1
    assert result[0].shape[-1] == target_T // 2 // extractor.model.downsample


def test_forward_htsat_windowed_concatenates_and_crops_padding():
    extractor = AudioFeatureExtractor.__new__(AudioFeatureExtractor)
    downsample = 16
    extractor.model = _FakeHTSAT(downsample=downsample, channels=4)

    def fake_single_window(spectrogram):
        time_len = spectrogram.shape[2] // downsample
        return [torch.arange(time_len).float().reshape(1, 1, 1, time_len)]

    extractor._forward_htsat_single_window = fake_single_window

    target_T = extractor.model.spec_size * extractor.model.freq_ratio  # 1024
    # 2.5 windows: two full windows plus a half-length tail that needs padding
    total_T = int(target_T * 2.5)
    long_clip = torch.zeros(1, 1, total_T, 64)

    result = AudioFeatureExtractor._forward_htsat_windowed(extractor, long_clip)

    assert len(result) == 1
    output_T = result[0].shape[-1]
    full_window_T = target_T // downsample
    # two full windows contribute their full length; the half window
    # contributes roughly half of a full window's output length
    expected_last = round(full_window_T * 0.5)
    assert output_T == 2 * full_window_T + expected_last


def test_forward_htsat_windowed_never_produces_an_empty_window_contribution():
    """Even a sliver of real audio in the final window must contribute >=1 step."""
    extractor = AudioFeatureExtractor.__new__(AudioFeatureExtractor)
    downsample = 16
    extractor.model = _FakeHTSAT(downsample=downsample, channels=4)

    def fake_single_window(spectrogram):
        time_len = spectrogram.shape[2] // downsample
        return [torch.arange(time_len).float().reshape(1, 1, 1, time_len)]

    extractor._forward_htsat_single_window = fake_single_window

    target_T = extractor.model.spec_size * extractor.model.freq_ratio
    total_T = target_T + 1  # one full window plus a single real frame
    clip = torch.zeros(1, 1, total_T, 64)

    result = AudioFeatureExtractor._forward_htsat_windowed(extractor, clip)

    full_window_T = target_T // downsample
    assert result[0].shape[-1] >= full_window_T + 1
