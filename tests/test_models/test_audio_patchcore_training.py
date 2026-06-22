import torch
from torch.utils.data import DataLoader, TensorDataset

from moviad.models.audio.patchcore.patchcore import PatchCore
from moviad.models.training_args import TrainingArgs


class FakeAudioFeatureExtractor:
    quantized = False

    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        return self

    def __call__(self, batch):
        batch_size = batch.shape[0]
        return [
            torch.randn(batch_size, 4, 8, 8, device=self.device),
            torch.randn(batch_size, 6, 4, 4, device=self.device),
        ]


def test_audio_patchcore_train_epoch_smoke():
    dataset = TensorDataset(torch.randn(4, 16000))
    dataloader = DataLoader(dataset, batch_size=2)

    model = PatchCore(
        device=torch.device("cpu"),
        input_size=(8, 8),
        feature_extractor=FakeAudioFeatureExtractor(),
        memory_bank_size=8,
        num_neighbors=1,
        blur=False,
    )

    result = model.train_epoch(
        epoch=0,
        train_dataloader=dataloader,
        training_args=TrainingArgs(batch_size=2, epochs=1),
    )

    assert result == 0.0
    assert model.memory_bank.numel() > 0
    assert model.memory_bank.ndim == 2

    model.eval()
    anomaly_maps, anomaly_scores = model(torch.randn(2, 16000))

    assert anomaly_maps.shape == (2, 1, 8, 8)
    assert anomaly_scores.shape == (2,)
