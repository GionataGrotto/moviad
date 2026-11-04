"""Self-supervised adaptation of BEATs to the machines, with LoRA, using the near and the far microphone.

The frozen BEATs of the patch-based detectors (PatchCore, PaDiM, CFA...) knows nothing about the machines. Here low-rank
adapters on the attention projections are trained on the normal training clips with one of three objectives, none of
which needs labels; the adapted network then replaces the frozen one as the feature extractor.

* ``reg``: the adapted network sees the **far** microphone and has to reproduce the token features that the frozen BEATs
  gives to the clean signal (``near``, or ``ls``: the near microphone after the least-squares removal of the far one).
* ``mask``: random 16x16 patches of the clean signal's filterbank are hidden; the adapted network has to reproduce the
  frozen features of the full signal at the hidden tokens.
* ``contrastive``: the clip embeddings of the far microphone and of the clean signal are pulled together within the batch
  (InfoNCE), so the representation keeps what the two microphones share.

The frozen network is the same model with the adapters switched off, so the hooks of the extractor serve both.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from moviad.datasets.dcase2026_task2 import ls_denoise
from moviad.models.audio.knn.finetune import LoRALinear, apply_lora

OBJECTIVES = ("reg", "mask", "contrastive")
SIGNALS = ("near", "ls")


def set_lora_active(model: torch.nn.Module, active: bool) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.active = active


@contextmanager
def lora_off(model: torch.nn.Module):
    """Run the block with the adapters switched off: the frozen network."""
    set_lora_active(model, False)
    try:
        yield
    finally:
        set_lora_active(model, True)


@contextmanager
def masked_input(model: torch.nn.Module, ratio: float, generator: torch.Generator, patch: tuple[int, int] = (16, 16)):
    """Hide random patches of the filterbank the model computes; yields a dict that receives the ``(B, time_patches, freq_patches)`` mask."""
    original = model.preprocess
    state: dict = {}

    def preprocess(*args, **kwargs):
        fbank = original(*args, **kwargs)  # (B, frames, bins)
        batch, frames, bins = fbank.shape
        mask = torch.rand(batch, frames // patch[0], bins // patch[1], generator=generator) < ratio
        state["mask"] = mask.to(fbank.device)
        full = state["mask"].repeat_interleave(patch[0], dim=1).repeat_interleave(patch[1], dim=2)
        full = F.pad(full, (0, bins - full.shape[2], 0, frames - full.shape[1]))
        return fbank.masked_fill(full, 0.0)  # the normalised filterbank has mean 0: hidden patches become the mean

    model.preprocess = preprocess
    try:
        yield state
    finally:
        model.__dict__.pop("preprocess", None)


def patch_size(model: torch.nn.Module) -> tuple[int, int]:
    kernel = getattr(getattr(model, "patch_embedding", None), "kernel_size", (16, 16))
    return (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)


def tokens(extractor, waveforms: torch.Tensor) -> list[torch.Tensor]:
    """The hooked blocks' tokens, one ``(B, N, D)`` tensor per requested layer (time-major, frequency fastest)."""
    extractor._captured.clear()
    extractor.model.extract_features(waveforms, padding_mask=torch.zeros_like(waveforms, dtype=torch.bool))
    return [extractor._captured[layer].transpose(0, 1) for layer in extractor.layers]


def token_loss(student: list[torch.Tensor], teacher: list[torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
    """1 - cosine similarity between the student and the teacher tokens, averaged over the layers (and over the masked tokens)."""
    losses = []
    for s, t in zip(student, teacher):
        distance = 1 - F.cosine_similarity(s, t.detach(), dim=-1)  # (B, N)
        losses.append(distance[mask].mean() if mask is not None else distance.mean())
    return torch.stack(losses).mean()


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def clean_signal(near: torch.Tensor, far: torch.Tensor, signal: str) -> torch.Tensor:
    if signal == "near":
        return near
    return torch.stack([ls_denoise(n, f) for n, f in zip(near, far)])


def crop_collate(crop_samples: int | None, generator: torch.Generator):
    """Collate ``[2, T]`` clips of different lengths (machines last 6, 10 or 16 s) into one ``[B, 2, W]`` batch.

    ``W`` is the length of the shortest clip, at most ``crop_samples``; every clip gives a random window of that length.
    """

    def collate(items: list[torch.Tensor]) -> torch.Tensor:
        window = min(item.shape[-1] for item in items)
        if crop_samples:
            window = min(window, crop_samples)
        starts = [int(torch.randint(0, item.shape[-1] - window + 1, (1,), generator=generator)) for item in items]
        return torch.stack([item[..., start : start + window] for item, start in zip(items, starts)])

    return collate


def adapt_beats(
    extractor,
    dataset,
    objective: str = "reg",
    signal: str = "near",
    epochs: int = 5,
    lr: float = 1e-4,
    rank: int = 8,
    alpha: float = 16.0,
    batch_size: int = 4,
    mask_ratio: float = 0.5,
    temperature: float = 0.1,
    seed: int = 0,
    crop_seconds: float | None = 4.0,
    sample_rate: int = 16000,
    log=print,
) -> list[float]:
    """Train LoRA adapters inside ``extractor.model`` (in place) and return the mean loss of every epoch.

    ``dataset`` yields ``[2, T]`` clips (near, far) at 16 kHz: ``DCASE2026Task2Dataset(..., channel="stereo")``.
    Memory: training keeps the activations of the 12 blocks of every clip, so each step sees a random window of
    ``crop_seconds`` (``None``: the whole clip) and the batch is small; the contrastive objective runs the network twice.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if signal not in SIGNALS:
        raise ValueError(f"signal must be one of {SIGNALS}, got {signal!r}")
    if len(dataset) < 2:
        raise ValueError("The self-supervised adaptation needs at least two clips")
    device = next(extractor.model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = extractor.model
    parameters = apply_lora(model, ("q_proj", "v_proj"), rank, alpha)
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    crop_samples = int(crop_seconds * sample_rate) if crop_seconds else None
    loader = DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, generator=generator, drop_last=len(dataset) > batch_size,
        collate_fn=crop_collate(crop_samples, generator),
    )
    patch = patch_size(model)
    model.eval()  # layerdrop and dropout would skip hooked blocks and add noise to the targets
    history = []
    for epoch in range(epochs):
        total, count = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            near, far = batch[:, 0], batch[:, 1]
            clean = clean_signal(near, far, signal)
            if objective == "reg":
                with torch.no_grad(), lora_off(model):
                    teacher = tokens(extractor, clean)
                loss = token_loss(tokens(extractor, far), teacher)
            elif objective == "mask":
                with torch.no_grad(), lora_off(model):
                    teacher = tokens(extractor, clean)
                with masked_input(model, mask_ratio, generator, patch) as state:
                    student = tokens(extractor, clean)
                mask = state["mask"]  # (B, time_patches, freq_patches) -> (B, N), same order as the tokens
                if not mask.any():
                    continue
                loss = token_loss(student, teacher, mask.flatten(1))
            else:
                za = tokens(extractor, far)[-1].mean(dim=1)
                zb = tokens(extractor, clean)[-1].mean(dim=1)
                loss = info_nce(za, zb, temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total, count = total + float(loss), count + 1
        history.append(total / max(count, 1))
        log(f"BEATs SSL ({objective}, {signal}) epoch {epoch + 1}/{epochs}: loss {history[-1]:.4f}")
    for parameter in parameters:
        parameter.requires_grad = False
    return history


def adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if "lora_" in name}


def save_adapter(extractor, path: str | Path, **meta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": adapter_state(extractor.model), "meta": meta}, path)


def load_adapter(extractor, path: str | Path, rank: int = 8, alpha: float = 16.0) -> None:
    """Add LoRA layers to ``extractor.model`` and load the weights saved by :func:`save_adapter`."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    meta = saved.get("meta", {})
    apply_lora(extractor.model, ("q_proj", "v_proj"), int(meta.get("rank", rank)), float(meta.get("alpha", alpha)))
    missing = extractor.model.load_state_dict(saved["state"], strict=False)
    unexpected = [k for k in missing.unexpected_keys]
    if unexpected or not saved["state"]:
        raise ValueError(f"The adapter {path} does not match this BEATs: unexpected keys {unexpected[:3]}")
    for name, parameter in extractor.model.named_parameters():
        parameter.requires_grad = False
