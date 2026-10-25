"""LoRA + ArcFace fine-tuning of a BEATs encoder on the normal training clips.

The machine attributes in the file names (``car_B1_spd_3_mic_2``...) are the only labels of the
challenge: they are used as classes of an additive-angular-margin (ArcFace) classifier on the
clip embedding, while only low-rank adapters on the attention projections are trained. The adapted
encoder then feeds the same nearest-neighbour detector as the frozen one. Machines whose clips all
share one attribute (``noAttribute``) have nothing to classify, so they are left unadapted.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from moviad.models.audio.knn.knn import AudioKNN


class LoRALinear(nn.Module):
    """``base(x) + (alpha / rank) * B(A(x))`` with the base layer frozen and B initialised to zero."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_b(self.lora_a(x))


def apply_lora(model: nn.Module, targets=("q_proj", "v_proj"), rank: int = 8, alpha: float = 16.0) -> list[nn.Parameter]:
    """Wrap every ``nn.Linear`` called one of ``targets``; return the trainable adapter parameters."""
    replaced = []
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                wrapped = LoRALinear(child, rank, alpha).to(child.weight.device)
                setattr(module, name, wrapped)
                replaced.append(wrapped)
    if not replaced:
        raise ValueError(f"No nn.Linear named {targets} found in the encoder: set finetune_targets in the config")
    return [p for layer in replaced for p in (*layer.lora_a.parameters(), *layer.lora_b.parameters())]


class ArcFaceHead(nn.Module):
    """Cosine classifier with an additive angular margin on the target class."""

    def __init__(self, dim: int, classes: int, margin: float = 0.3, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(classes, dim) * 0.01)
        self.margin, self.scale = margin, scale

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-6, 1 - 1e-6)
        target = torch.cos(torch.acos(cosine) + self.margin)
        one_hot = F.one_hot(labels, cosine.shape[1]).bool()
        return F.cross_entropy(self.scale * torch.where(one_hot, target, cosine), labels)


class FineTunedKNN(AudioKNN):
    """AudioKNN whose BEATs encoder is first adapted with LoRA + ArcFace on the attribute labels."""

    def __init__(self, encoder, device, top_k=1, metric="cosine", rank=8, alpha=16.0, targets=("q_proj", "v_proj"),
                 epochs=10, lr=1e-4, batch_size=8, margin=0.3, scale=16.0, seed=0):
        super().__init__(encoder, device, top_k=top_k, metric=metric)
        self.lora_params = apply_lora(encoder.model, targets, rank, alpha)
        self.hyper = dict(epochs=epochs, lr=lr, batch_size=batch_size, margin=margin, scale=scale, seed=seed)

    def _embed_grad(self, waveforms: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        waveforms = waveforms.to(self.device)
        encoder._captured.clear()
        encoder.model.extract_features(waveforms, padding_mask=torch.zeros_like(waveforms, dtype=torch.bool))
        last = encoder._captured[encoder.layers[-1]]  # [T, B, D]
        return last.transpose(0, 1).mean(dim=1)

    def finetune(self, waveforms: torch.Tensor, labels: list[str]) -> float | None:
        classes = sorted(set(labels))
        if len(classes) < 2:
            print("finetune: a single attribute class, the encoder is left unadapted", flush=True)
            return None
        index = torch.tensor([classes.index(label) for label in labels])
        h = self.hyper
        torch.manual_seed(h["seed"])
        with torch.no_grad():
            dim = self._embed_grad(waveforms[:1]).shape[-1]
        head = ArcFaceHead(dim, len(classes), h["margin"], h["scale"]).to(self.device)
        optimizer = torch.optim.AdamW([*self.lora_params, *head.parameters()], lr=h["lr"])
        # stay in eval mode: BEATs' layerdrop would skip random blocks (and their hooks) in train mode
        self.encoder.model.eval()
        loss_value = None
        for _ in range(h["epochs"]):
            order = torch.randperm(len(waveforms))
            for start in range(0, len(order), h["batch_size"]):
                batch = order[start : start + h["batch_size"]]
                loss = head(self._embed_grad(waveforms[batch]), index[batch].to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_value = float(loss)
        self.encoder.model.eval()
        return loss_value

    def fit(self, train_dataloader) -> None:
        dataset = train_dataloader.dataset
        records = getattr(dataset, "records", None)
        if records is None:
            raise ValueError("FineTunedKNN needs a dataset with .records carrying the attribute of every clip")
        waveforms = torch.stack([dataset[i] for i in range(len(dataset))])
        self.finetune(waveforms, [record.attribute for record in records])
        super().fit(train_dataloader)
