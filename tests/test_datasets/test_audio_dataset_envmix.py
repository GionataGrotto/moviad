"""Contract test for the EnvMix builder, with stubbed source datasets.

Keeps the shapes returned by ``generate_urban_esc_V1`` pinned without needing
UrbanSound8K or ESC-50 on disk.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class FakeUrbanSound(Dataset):
    """Two four second background clips at the Cnn14 sample rate."""

    def __init__(self, *args, **kwargs):
        self.items = [torch.randn(44100 * 4), torch.randn(44100 * 4)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index], 44100


class FakeEsc50(Dataset):
    """Two one second anomaly events of a single category."""

    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 2

    def get_with_category(self, index):
        return torch.randn(44100), 44100, "glass_breaking"


def test_generate_urban_esc_v1_returns_the_expected_splits(monkeypatch):
    import moviad.datasets.audio_dataset as audio_dataset
    from moviad.utilities.audio.audio_feature_extractor import AudioFeatureExtractor

    _, _, transform = AudioFeatureExtractor._load_spectrogram_transform("Cnn14")

    monkeypatch.setattr(audio_dataset, "UrbanSound8KDataset", FakeUrbanSound)
    monkeypatch.setattr(audio_dataset, "Esc50Dataset", FakeEsc50)

    train_dataset, test_dataset, test_dataset_ff = audio_dataset.generate_urban_esc_V1(
        "air_conditioner",
        ["glass_breaking"],
        transform,
        SNR_dB=6.0,
        # generate_urban_esc_V1 probes the roots with Path.exists()
        path_urban=Path("."),
        path_esc50=Path("."),
        seed=42,
        max_num_samples=2,
        target_duration=4.0,
    )

    assert len(train_dataset) > 0
    assert len(test_dataset) > 0
    assert len(test_dataset_ff) > 0

    waveform, label, gt_mask, gt_temporal_mask, path = test_dataset[0]

    assert isinstance(waveform, torch.Tensor)
    assert waveform.ndim >= 1
    assert int(label) in (0, 1)
    assert gt_mask.ndim >= 2
    assert gt_temporal_mask.ndim >= 1
    assert isinstance(path, str)
