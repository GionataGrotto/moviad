"""BEATs (Microsoft unilm) as a feature extractor for the patch-based anomaly detectors.

BEATs is a ViT over a 128-bin Kaldi filterbank: a 16x16 patch embedding turns the
``(frames, 128)`` spectrogram into ``frames/16`` x ``128/16 = 8`` tokens (time-major, frequency
fastest). Hooking the output of transformer blocks and folding the tokens back onto that grid
gives ``(B, 768, frames/16, 8)`` feature maps, the same kind of input PatchCore, PaDiM and CFA
consume from Cnn14.

The BEATs source (``BEATs.py`` and its siblings) and a checkpoint such as
``BEATs_iter3_plus_AS2M.pt`` are not part of this repository. Audio must be 16 kHz.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn


def import_beats(code_dir):
    """Import ``BEATs`` and ``BEATsConfig`` from ``code_dir``.

    Two layouts exist: the unilm one (a plain folder, ``import BEATs``) and the one where the
    folder is a package whose files import each other as ``from beats.backbone import ...`` (it
    has an ``__init__.py``). The latter is imported as ``<folder name>.BEATs`` with the parent
    folder on ``sys.path``.
    """
    module_name = "BEATs"
    if code_dir is not None:
        folder = Path(code_dir).expanduser()
        if (folder / "__init__.py").exists():
            root, module_name = str(folder.parent), f"{folder.name}.BEATs"
        else:
            root = str(folder)
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name not in (module_name, module_name.split(".")[0]) or code_dir is None:
            raise  # BEATs was found but one of its own imports is missing
        raise ModuleNotFoundError(
            f"Cannot find BEATs.py (knn_beats_code_dir={str(code_dir)!r}): set it to the folder that contains it"
        ) from error
    return module.BEATs, module.BEATsConfig


class BEATsFeatureExtractor(nn.Module):
    """Waveforms ``[B, T]`` (16 kHz) -> one ``(B, 768, time_patches, 8)`` map per requested block.

    Args:
        checkpoint_path: BEATs ``.pt`` checkpoint (``{"cfg": ..., "model": ...}``).
        code_dir: directory containing ``BEATs.py``; ``None`` if it is already importable.
        layers: transformer block numbers, 1-based (12 is the last block of BEATs-base).
        device: device used for the forward pass.
        frozen: keep the network in eval mode without gradients (teacher / plain extractor).
        pretrained: load the checkpoint weights; ``False`` keeps a random initialisation with
            the checkpoint's architecture (e.g. for a student network).
    """

    sample_rate = 16000
    MEL_BINS = 128
    FRAME_LENGTH = 400  # 25 ms at 16 kHz
    FRAME_SHIFT = 160  # 10 ms at 16 kHz

    def __init__(self, checkpoint_path, code_dir=None, layers=(8, 10, 12), device="cpu", frozen=True, pretrained=True):
        super().__init__()
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"BEATs checkpoint not found: {checkpoint_path}")
        BEATs, BEATsConfig = import_beats(code_dir)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model = BEATs(BEATsConfig(checkpoint["cfg"]))
        if pretrained:
            self.model.load_state_dict(checkpoint["model"])

        blocks = self.model.encoder.layers
        self.layers = tuple(int(layer) for layer in layers)
        if not self.layers or any(not 1 <= layer <= len(blocks) for layer in self.layers):
            raise ValueError(f"layers must be block numbers in [1, {len(blocks)}], got {self.layers}")
        self._captured: dict[int, torch.Tensor] = {}
        for layer in self.layers:
            blocks[layer - 1].register_forward_hook(self._make_hook(layer))

        self.frozen = frozen
        self.spec_shape = None
        self.device = torch.device("cpu")
        self.train(not frozen)
        if frozen:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
        self.to(device)

    def _make_hook(self, layer):
        def hook(_module, _inputs, output):
            self._captured[layer] = output[0] if isinstance(output, (tuple, list)) else output

        return hook

    @classmethod
    def fbank_shape(cls, num_samples: int) -> tuple[int, int]:
        """``(frames, mel bins)`` of the Kaldi filterbank BEATs computes for ``num_samples`` samples."""
        return 1 + (num_samples - cls.FRAME_LENGTH) // cls.FRAME_SHIFT, cls.MEL_BINS

    def _frequency_patches(self) -> int:
        kernel = self.model.patch_embedding.kernel_size
        patch = kernel[1] if isinstance(kernel, (tuple, list)) else kernel
        return self.MEL_BINS // patch

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.train(mode and not self.frozen)
        return self

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def forward(self, waveforms: torch.Tensor) -> list[torch.Tensor]:
        waveforms = waveforms.to(self.device)
        padding_mask = torch.zeros_like(waveforms, dtype=torch.bool)
        self._captured.clear()
        with torch.no_grad() if self.frozen else nullcontext():
            self.model.extract_features(waveforms, padding_mask=padding_mask)

        freq_patches = self._frequency_patches()
        maps = []
        for layer in self.layers:
            tokens = self._captured[layer]  # blocks are time-first: [tokens, B, D], index = time * freq + band
            count, batch, dim = tokens.shape
            if count % freq_patches:
                raise RuntimeError(f"{count} tokens do not fold onto {freq_patches} frequency patches")
            maps.append(tokens.permute(1, 2, 0).reshape(batch, dim, count // freq_patches, freq_patches))
        self.spec_shape = torch.Size([waveforms.shape[0], 1, *self.fbank_shape(waveforms.shape[-1])])
        return maps
