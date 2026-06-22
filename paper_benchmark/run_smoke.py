from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from benchmark_common import PROJECT_ROOT
from benchmark_common import run_cli


class TinyAudioDataset(Dataset):
    def __init__(self, train: bool):
        self.train = train

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        waveform = torch.randn(44100)
        if self.train:
            return waveform
        label = torch.tensor(index % 2, dtype=torch.long)
        gt_mask = torch.zeros(1, 8, 8)
        gt_tmp_mask = torch.zeros(8)
        return waveform, label, gt_mask, gt_tmp_mask, f"synthetic_{index}.wav"


def main() -> None:
    import moviad
    from moviad.models.audio.patchcore.patchcore import PatchCore
    from moviad.models.audio.audio_vad_model import AudioVADModel

    assert PROJECT_ROOT.exists()
    assert moviad is not None
    assert issubclass(PatchCore, AudioVADModel)

    train_loader = DataLoader(TinyAudioDataset(train=True), batch_size=2)
    test_loader = DataLoader(TinyAudioDataset(train=False), batch_size=2)
    first_train_batch = next(iter(train_loader))
    first_test_batch = next(iter(test_loader))

    assert first_train_batch.shape == (2, 44100)
    assert len(first_test_batch) == 5
    print("Smoke OK: package imports and audio dataloader shape are valid.")


if __name__ == "__main__":
    run_cli(main)
