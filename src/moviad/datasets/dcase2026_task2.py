"""Dataset utilities for the DCASE 2026 Task 2 development dataset.

The official dataset encodes the split and domain in each WAV filename.  This
loader deliberately selects channel 0 (the near microphone) and keeps the
metadata needed to produce DCASE evaluator-compatible files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


FILENAME_RE = re.compile(
    r"^section_(?P<section>\d+)_(?P<domain>source|target)_"
    r"(?P<split>train|test)_(?P<label>normal|anomaly)_.*\.wav$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DCASERecord:
    path: Path
    machine_type: str
    section: str
    domain: str
    split: str
    label: int


def parse_dcase_filename(path: Path, machine_type: str) -> DCASERecord:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Not a DCASE 2026 Task 2 filename: {path.name}")
    groups = match.groupdict()
    return DCASERecord(
        path=path,
        machine_type=machine_type,
        section=f"{int(groups['section']):02d}",
        domain=groups["domain"].lower(),
        split=groups["split"].lower(),
        label=int(groups["label"].lower() == "anomaly"),
    )


def discover_machine_types(root: str | Path) -> list[str]:
    root = Path(root).expanduser()
    return sorted(path.name for path in root.iterdir() if path.is_dir())


class DCASE2026Task2Dataset(Dataset):
    """One machine/section of DCASE 2026 Task 2.

    Parameters
    ----------
    root:
        Directory containing the seven machine directories.
    machine_type:
        Directory name, for example ``fan`` or ``ToyCarEmu``.
    split:
        ``train`` or ``test``.
    domain:
        ``all`` (default), ``source`` or ``target``.
    train_domains:
        For a training dataset, ``all`` uses the official 990+10 normal
        protocol.  ``source`` and ``target`` are useful diagnostic variants.
    """

    def __init__(
        self,
        root: str | Path,
        machine_type: str,
        split: str,
        domain: str = "all",
        train_domains: str = "all",
        channel: int = 0,
        target_sample_rate: int = 44100,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.machine_type = machine_type
        self.split = split.lower()
        self.domain = domain.lower()
        self.train_domains = train_domains.lower()
        self.channel = int(channel)
        self.target_sample_rate = int(target_sample_rate)
        self._resampler = torchaudio.transforms.Resample(16000, self.target_sample_rate)
        if self.split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if self.domain not in {"all", "source", "target"}:
            raise ValueError("domain must be 'all', 'source' or 'target'")
        if self.train_domains not in {"all", "source", "target"}:
            raise ValueError("train_domains must be 'all', 'source' or 'target'")

        machine_root = self.root / machine_type
        if not machine_root.is_dir():
            raise FileNotFoundError(f"Machine directory not found: {machine_root}")
        records = []
        for path in sorted(machine_root.rglob("*.wav")):
            try:
                record = parse_dcase_filename(path, machine_type)
            except ValueError:
                continue
            if record.split != self.split:
                continue
            requested_domain = self.train_domains if self.split == "train" else self.domain
            if requested_domain != "all" and record.domain != requested_domain:
                continue
            if self.split == "train" and record.label != 0:
                continue
            records.append(record)
        self.records = records
        if not self.records:
            raise RuntimeError(
                f"No DCASE files found for machine={machine_type}, split={split}, "
                f"domain={domain}, train_domains={train_domains}"
            )

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        waveform, _ = torchaudio.load(str(path))
        return waveform.to(torch.float32)

    def __getitem__(self, index: int):
        record = self.records[index]
        waveform = self._load(record.path)
        if waveform.ndim != 2 or self.channel >= waveform.shape[0]:
            raise ValueError(f"Expected channel {self.channel} in stereo WAV: {record.path}")
        # Baseline: near microphone only. Keep the waveform shape [samples].
        waveform = waveform[self.channel]
        if self.target_sample_rate != 16000:
            waveform = self._resampler(waveform.unsqueeze(0)).squeeze(0)
        if self.split == "train":
            return waveform
        # Strings are intentionally returned instead of the dataclass: PyTorch's
        # default collate function can batch them, while the record list remains
        # available through ``self.records`` for metadata-aware evaluation.
        return waveform, record.label, str(record.path)


__all__ = [
    "DCASE2026Task2Dataset",
    "DCASERecord",
    "discover_machine_types",
    "parse_dcase_filename",
]
