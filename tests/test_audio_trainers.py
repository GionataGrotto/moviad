import importlib
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from moviad.models.audio.padim.padim import Padim
from moviad.models.audio.patchcore.patchcore import PatchCore
from moviad.trainers.audio.trainer_padim import PadimTrainer
from moviad.trainers.audio.trainer_patchcore import TrainerPatchCore


class Backbone(nn.Module):
    """Waveform [B, 256] -> Cnn14-like maps (128/256/512 channels)."""

    def forward(self, x):
        base = x.reshape(x.shape[0], 1, 16, 16)
        maps = []
        for channels, pool in ((128, 1), (256, 2), (512, 4)):
            pooled = torch.nn.functional.avg_pool2d(base, pool) if pool > 1 else base
            maps.append(torch.sin(pooled * torch.linspace(0.5, 2, channels).view(1, -1, 1, 1)))
        return maps


def labelled_loader():
    # (waveform, label) batches, as produced by the DCASE / MIMII test splits
    return DataLoader(TensorDataset(torch.randn(8, 256), torch.zeros(8)), batch_size=4)


def test_padim_trainer_fits_and_saves_from_tuple_batches(tmp_path):
    model = Padim(
        "Cnn14", "ToyCar", torch.device("cpu"), ["conv_block2", "conv_block3", "conv_block4"],
        diag_cov=True, img_size=(16, 16), backbone_model=Backbone(), embedding_dim=8,
    )
    trainer = PadimTrainer(model, "cpu", str(tmp_path), None, "ToyCar")

    trainer.train(labelled_loader(), streaming=False)  # the flag is accepted and ignored

    assert model.gauss_mean is not None
    assert (tmp_path / "checkpoints_Cnn14" / "train_ToyCar.pth.tar").exists()


def test_patchcore_trainer_fits_from_tuple_batches():
    class Extractor:
        def to(self, device):
            return self

        def __call__(self, batch):
            return Backbone()(batch)[:2]

    model = PatchCore(torch.device("cpu"), (16, 16), Extractor(), memory_bank_size=40, num_neighbors=1)
    trainer = TrainerPatchCore(model, labelled_loader(), None, "cpu", force_cpu=True)

    trainer.train(streaming=True)  # the flag is accepted and ignored

    assert model.memory_bank.shape == (40, 128 + 256)


def test_audio_trainers_import_without_wandb(monkeypatch):
    """wandb is only needed when logging; a missing install must not break the import."""
    monkeypatch.setitem(sys.modules, "wandb", None)  # `import wandb` now raises ImportError
    for name in ("moviad.trainers.audio.trainer_cfa", "moviad.trainers.audio.trainer_stfpm"):
        sys.modules.pop(name, None)
        importlib.import_module(name)
