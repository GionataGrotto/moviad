import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from moviad.trainers.audio.trainer_stfpm import build_optimizer


def test_sgd_keeps_the_paper_setup_and_adamw_is_available():
    parameters = nn.Linear(3, 3).parameters()
    sgd = build_optimizer(parameters, "sgd", 0.4)
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.param_groups[0]["lr"] == 0.4 and sgd.param_groups[0]["momentum"] == 0.9

    adamw = build_optimizer(nn.Linear(3, 3).parameters(), "adamw", 1e-5, 1e-2)
    assert isinstance(adamw, torch.optim.AdamW)
    assert adamw.param_groups[0]["lr"] == 1e-5 and adamw.param_groups[0]["weight_decay"] == 1e-2


def test_unknown_optimizer_is_rejected():
    with pytest.raises(ValueError, match="Unsupported STFPM optimizer"):
        build_optimizer(nn.Linear(3, 3).parameters(), "rmsprop")


def test_trainer_accepts_batches_that_carry_labels(monkeypatch):
    """The DCASE test split yields (waveform, label, path); the trainer used to call .to() on the tuple."""
    from moviad.models.audio.stfpm.stfpm import STFPM
    from moviad.trainers.audio.trainer_stfpm import TrainerSTFPM

    class Extractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Conv2d(1, 4, 3, padding=1)
            self.device = torch.device("cpu")
            self.spec_shape = (1, 1, 8, 8)

        def forward(self, x):
            return [self.model(x.reshape(x.shape[0], 1, 8, 8))]

        def to(self, device):
            return self

    teacher, student = Extractor(), Extractor()
    loader = DataLoader(TensorDataset(torch.randn(4, 64), torch.zeros(4)), batch_size=2)
    trainer = TrainerSTFPM(STFPM(teacher, student), loader, None, wandb=False, device="cpu", optimizer="sgd", learning_rate=0.01)

    trainer.train(1, [], None)  # no crash on the (waveform, label) batches
