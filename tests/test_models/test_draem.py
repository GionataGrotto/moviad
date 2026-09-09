import torch
from torch.utils.data import Dataset


IMG_SIZE = 64


class FakeVADDataset(Dataset):
    """Minimal stand-in for a VAD dataset, so the test needs no data on disk."""

    def __init__(self, length: int, split: str):
        self.length = length
        self.split = split
        generator = torch.Generator().manual_seed(0)
        self.images = torch.rand(length, 3, IMG_SIZE, IMG_SIZE, generator=generator)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = self.images[index]
        if self.split == "train":
            return image

        label = index % 2
        mask = torch.zeros(1, IMG_SIZE, IMG_SIZE, dtype=torch.int32)
        if label:
            mask[:, 8:24, 8:24] = 1
        return image, label, mask, f"fake/{index}.png"


def build_model():
    from moviad.models.draem.draem import DRAEM

    return DRAEM(
        input_size=(IMG_SIZE, IMG_SIZE),
        base_width_reconstructive=4,
        base_width_discriminative=4,
    )


def test_anomaly_generator_contract():
    from moviad.models.components.draem.anomaly_generator import DraemAnomalyGenerator

    torch.manual_seed(0)
    generator = DraemAnomalyGenerator(anomaly_probability=1.0)
    images = torch.rand(4, 3, IMG_SIZE, IMG_SIZE)

    augmented, masks, labels = generator(images)

    assert augmented.shape == images.shape
    assert masks.shape == (4, 1, IMG_SIZE, IMG_SIZE)
    assert labels.shape == (4,)
    assert augmented.min() >= 0.0 and augmented.max() <= 1.0
    assert set(masks.unique().tolist()).issubset({0.0, 1.0})
    # the image is left untouched wherever the anomaly mask is zero
    assert torch.allclose(augmented * (1 - masks), images * (1 - masks), atol=1e-6)


def test_model_forward_and_train_step():
    from moviad.models.draem.draem import DRAEMTrainArgs

    torch.manual_seed(0)
    model = build_model().to(torch.device("cpu"))
    images = torch.rand(2, 3, IMG_SIZE, IMG_SIZE)

    training_args = DRAEMTrainArgs(batch_size=2, epochs=2)
    training_args.init_train(model)

    model.train()
    params_before = [p.clone() for p in model.parameters()]
    loss = model.train_step(images, training_args)
    params_after = list(model.parameters())

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))

    model.eval()
    with torch.no_grad():
        anomaly_maps, anomaly_scores = model(images)

    assert anomaly_maps.shape == (2, 1, IMG_SIZE, IMG_SIZE)
    assert anomaly_scores.shape == (2,)
    assert anomaly_maps.min() >= 0.0 and anomaly_maps.max() <= 1.0


def test_model_create_train():
    from moviad.models.draem.draem import DRAEMTrainArgs
    from moviad.trainers.trainer import Trainer
    from moviad.utilities.evaluation.metrics import AvgPrec, MetricLvl, RocAuc

    torch.manual_seed(0)
    device = torch.device("cpu")

    model = build_model().to(device)
    training_args = DRAEMTrainArgs(batch_size=2, epochs=2)

    trainer = Trainer(
        training_args,
        model,
        FakeVADDataset(4, "train"),
        FakeVADDataset(4, "test"),
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
        ],
        device=device,
        logger=None,
        save_path=None,
        saving_criteria=None,
    )
    trainer.train_dataloader = torch.utils.data.DataLoader(
        FakeVADDataset(4, "train"), batch_size=2, shuffle=True, num_workers=0
    )
    trainer.eval_dataloader = torch.utils.data.DataLoader(
        FakeVADDataset(4, "test"), batch_size=2, shuffle=False, num_workers=0
    )

    params_before = [p.clone() for p in model.parameters()]
    trainer.train()
    params_after = list(model.parameters())

    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
