"""Frozen pretrained audio encoders returning one clip-level embedding per layer."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


class BEATsEncoder:
    """BEATs (Microsoft unilm) encoder; embeddings are the time-average of transformer blocks.

    Needs two files that are not part of this repository: the BEATs source
    (``BEATs.py`` and its siblings, e.g. the ``beats/`` folder of unilm) and a
    checkpoint such as ``BEATs_iter3_plus_AS2M.pt``. Audio must be 16 kHz.

    Args:
        checkpoint_path: BEATs ``.pt`` checkpoint (``{"cfg": ..., "model": ...}``).
        code_dir: directory containing ``BEATs.py``; ``None`` if it is already importable.
        layers: transformer block numbers, 1-based (12 is the last block of BEATs-base).
        device: device used for the forward pass.
    """

    sample_rate = 16000

    def __init__(self, checkpoint_path, code_dir=None, layers=(10,), device="cpu"):
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"BEATs checkpoint not found: {checkpoint_path}")
        BEATs, BEATsConfig = self._import_beats(code_dir)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model = BEATs(BEATsConfig(checkpoint["cfg"]))
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        blocks = self.model.encoder.layers
        self.layers = tuple(int(layer) for layer in layers)
        if not self.layers or any(not 1 <= layer <= len(blocks) for layer in self.layers):
            raise ValueError(f"layers must be block numbers in [1, {len(blocks)}], got {self.layers}")
        self._captured: dict[int, torch.Tensor] = {}
        for layer in self.layers:
            blocks[layer - 1].register_forward_hook(self._make_hook(layer))
        self.device = torch.device("cpu")
        self.to(device)

    @staticmethod
    def _import_beats(code_dir):
        if code_dir is not None:
            code_dir = str(Path(code_dir).expanduser())
            if code_dir not in sys.path:
                sys.path.insert(0, code_dir)
        try:
            module = importlib.import_module("BEATs")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Cannot import BEATs.py: set knn_beats_code_dir to the folder that contains it"
            ) from error
        return module.BEATs, module.BEATsConfig

    def _make_hook(self, layer):
        def hook(_module, _inputs, output):
            self._captured[layer] = output[0] if isinstance(output, (tuple, list)) else output

        return hook

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def __call__(self, waveforms: torch.Tensor) -> list[torch.Tensor]:
        """``[B, T]`` 16 kHz waveforms -> one ``[B, D]`` embedding per requested layer."""
        waveforms = waveforms.to(self.device)
        padding_mask = torch.zeros_like(waveforms, dtype=torch.bool)
        self._captured.clear()
        with torch.no_grad():
            self.model.extract_features(waveforms, padding_mask=padding_mask)
        # BEATs transformer blocks are time-first: [T, B, D]
        return [self._captured[layer].transpose(0, 1).mean(dim=1) for layer in self.layers]
