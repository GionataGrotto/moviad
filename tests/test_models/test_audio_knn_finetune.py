import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from moviad.datasets.dcase2026_task2 import DCASERecord
from moviad.models.audio.knn.finetune import ArcFaceHead, FineTunedKNN, LoRALinear, apply_lora


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.v_proj, self.k_proj = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8)

    def forward(self, x):
        return torch.tanh(self.q_proj(x) + self.v_proj(x) + self.k_proj(x))


class ToyBEATs(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
        self.embed = nn.Linear(4, 8)
        self.hook_target = self.block
        self.captured = None

    def extract_features(self, waveforms, padding_mask=None):
        x = self.embed(waveforms.view(waveforms.shape[0], -1, 4)).transpose(0, 1)  # [T, B, D]
        return self.block(x)


class ToyEncoder:
    layers = (1,)

    def __init__(self):
        self.model = ToyBEATs()
        for p in self.model.parameters():
            p.requires_grad = False
        self._captured = {}
        self.model.block.register_forward_hook(lambda _m, _i, out: self._captured.__setitem__(1, out))

    def to(self, device):
        return self

    def __call__(self, waveforms):
        self._captured.clear()
        with torch.no_grad():
            self.model.extract_features(waveforms)
        return [self._captured[1].transpose(0, 1).mean(dim=1)]


class ToyDataset(Dataset):
    def __init__(self, attributes):
        generator = torch.Generator().manual_seed(0)
        self.clips = [torch.randn(16, generator=generator) + (3.0 if a == "B" else 0.0) for a in attributes]
        self.records = [DCASERecord(None, "m", "00", "source", "train", 0, a) for a in attributes]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return self.clips[i]


def test_lora_starts_as_identity_and_only_adapters_train():
    block = Block()
    x = torch.randn(3, 8)
    before = block(x)
    params = apply_lora(block, ("q_proj", "v_proj"), rank=2)
    assert isinstance(block.q_proj, LoRALinear) and isinstance(block.k_proj, nn.Linear)
    assert torch.allclose(block(x), before, atol=1e-6) and len(params) == 4


def test_arcface_margin_raises_the_loss():
    embeddings, labels = torch.randn(6, 8), torch.tensor([0, 1, 0, 1, 0, 1])
    assert ArcFaceHead(8, 2, margin=0.5)(embeddings, labels) > ArcFaceHead(8, 2, margin=0.0)(embeddings, labels) - 1e-6


def test_finetune_updates_adapters_and_single_attribute_is_skipped():
    dataset = ToyDataset(["A", "B"] * 6)
    model = FineTunedKNN(ToyEncoder(), torch.device("cpu"), targets=("q_proj", "v_proj"), epochs=3, batch_size=4)
    before = [p.clone() for p in model.lora_params]
    model.fit(DataLoader(dataset, batch_size=4))
    assert any(not torch.equal(a, b) for a, b in zip(before, model.lora_params))
    assert model.memory[0].shape[0] == 12

    flat = FineTunedKNN(ToyEncoder(), torch.device("cpu"), epochs=3)
    before = [p.clone() for p in flat.lora_params]
    flat.fit(DataLoader(ToyDataset(["noAttribute"] * 6), batch_size=3))
    assert all(torch.equal(a, b) for a, b in zip(before, flat.lora_params))


def test_lora_exposes_the_adapted_weight_for_code_that_reads_it_directly():
    layer = LoRALinear(nn.Linear(8, 8), rank=2)
    nn.init.normal_(layer.lora_b.weight)
    x = torch.randn(3, 8)
    direct = torch.nn.functional.linear(x, layer.weight, layer.bias)
    assert torch.allclose(direct, layer(x), atol=1e-5)
    direct.sum().backward()
    assert layer.lora_a.weight.grad is not None


def test_pseudo_classes_make_machines_without_attributes_trainable():
    model = FineTunedKNN(ToyEncoder(), torch.device("cpu"), epochs=3, batch_size=4, pseudo_classes=2)
    before = [p.clone() for p in model.lora_params]
    model.fit(DataLoader(ToyDataset(["noAttribute"] * 6 + ["noAttribute"] * 6), batch_size=4))
    assert any(not torch.equal(a, b) for a, b in zip(before, model.lora_params))
