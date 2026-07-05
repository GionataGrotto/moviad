from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from scipy.io import wavfile

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.configurations import LabelName, Split


_VALID_SPLIT_NAMES = {Split.TRAIN.value, Split.TEST.value, Split.VAL.value, "validation", "val"}


class MIMIDataset(VADDataset):
    """Dataset wrapper for the MIMII benchmark.

    The loader accepts both the older split-less layout used by the downloaded
    archive and the newer pre-split layouts. When the dataset has no explicit
    train/test folders, we create a deterministic split from the available WAV
    files.
    """

    CATEGORIES = ["fan", "pump", "slider", "valve"]

    def __init__(
        self,
        dataset_directory: str,
        snr: str,
        category: str,
        machine_id: str,
        split: Split,
        wave_to_spectro=None,
        transform=None,
        seed: int = 42,
        img_size: tuple[int, int] | None = None,
    ):
        self.dataset_directory = Path(dataset_directory)
        self.snr = snr
        self.machine_id = machine_id
        self.wave_to_spectro = wave_to_spectro
        self.transform = transform
        self.seed = seed
        self.img_size = tuple(img_size) if img_size is not None else None

        args = DatasetArguments(
            dataset_path=str(self.dataset_directory),
            img_size=self.img_size or (1, 1),
            gt_mask_size=self.img_size or (1, 1),
        )
        super().__init__(args, category, split)

        self.dataset_df: pd.DataFrame | None = None
        self._mask_size = tuple(self.dataset_arguments.gt_mask_size)
        self._load_directory = self._resolve_load_directory()

        self._load_dataset()
        self._infer_mask_size()

    def is_loaded(self) -> bool:
        return self.dataset_df is not None

    def split_dataset(self, train_size, valid_size):
        return None

    @staticmethod
    def get_categories() -> list:
        return list(MIMIDataset.CATEGORIES)

    def contaminate(self, ratio: float, seed: int = 42) -> int:
        return 0

    def compute_contamination_ratio(self) -> float:
        return 0.0

    @staticmethod
    def _load_waveform(file_path: str) -> tuple[torch.Tensor, int]:
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            return waveform, sample_rate
        except Exception:
            sample_rate, data = wavfile.read(file_path)
            waveform = torch.from_numpy(data)

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)

            if waveform.dtype.is_floating_point:
                waveform = waveform.to(torch.float32)
            else:
                max_val = float(torch.iinfo(waveform.dtype).max)
                waveform = waveform.to(torch.float32) / max_val

            return waveform, int(sample_rate)

    def _resolve_load_directory(self) -> Path:
        candidates: list[Path] = []

        if self.snr:
            candidates.append(self.dataset_directory / self.snr / self.category / self.machine_id)
            candidates.append(self.dataset_directory / self.snr / self.category)
        candidates.append(self.dataset_directory / self.category / self.machine_id)
        candidates.append(self.dataset_directory / self.category)
        if self.machine_id:
            candidates.append(self.dataset_directory / self.machine_id)
        candidates.append(self.dataset_directory)

        for candidate in candidates:
            if candidate.exists() and any(candidate.rglob("*.wav")):
                return candidate

        raise FileNotFoundError(
            f"Could not find WAV files for category={self.category}, machine_id={self.machine_id} "
            f"under {self.dataset_directory}"
        )

    def _load_dataset(self):
        load_directory = self._load_directory
        files = sorted(load_directory.rglob("*.wav"))

        if not files:
            raise FileNotFoundError(f"No WAV files found in: {load_directory}")

        records = []
        for file_path in files:
            relative = file_path.relative_to(load_directory)
            parts = relative.parts
            split_hint = parts[0].lower() if parts and parts[0].lower() in _VALID_SPLIT_NAMES else None
            machine_hint = next((part for part in parts if part.startswith("id_")), load_directory.name if load_directory.name.startswith("id_") else "")
            label_hint = parts[-2].lower() if len(parts) >= 2 else "normal"

            records.append(
                {
                    "file_path": str(file_path),
                    "machine_id": machine_hint,
                    "label_text": label_hint,
                    "split_hint": split_hint,
                }
            )

        dataset_df = pd.DataFrame(records)
        dataset_df["label"] = dataset_df["label_text"].map(
            lambda label: LabelName.NORMAL if label == "normal" else LabelName.ABNORMAL
        )

        if dataset_df["split_hint"].notna().any():
            split_map = {
                "train": Split.TRAIN.value,
                "test": Split.TEST.value,
                "valid": Split.VAL.value,
                "validation": Split.VAL.value,
                "val": Split.VAL.value,
            }
            dataset_df["split"] = dataset_df["split_hint"].map(split_map)
            self.dataset_df = self._select_split(dataset_df)
            return

        normal_samples = dataset_df[dataset_df["label"] == LabelName.NORMAL]
        abnormal_samples = dataset_df[dataset_df["label"] == LabelName.ABNORMAL]

        if len(abnormal_samples) == 0:
            raise RuntimeError(f"No abnormal samples found in {load_directory}")

        test_size = max(1, int(len(abnormal_samples) * 0.3))
        test_size = min(test_size, len(normal_samples), len(abnormal_samples))

        test_normal_samples = normal_samples.sample(n=test_size, random_state=self.seed, replace=False)
        test_abnormal_samples = abnormal_samples.sample(n=test_size, random_state=self.seed, replace=False)
        test_df = pd.concat([test_normal_samples, test_abnormal_samples])

        train_df = dataset_df.drop(test_df.index)

        test_df = test_df.reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)

        if self.split == Split.TRAIN:
            self.dataset_df = train_df[train_df["label"] == LabelName.NORMAL].reset_index(drop=True)
        elif self.split == Split.TEST:
            self.dataset_df = test_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        else:
            self.dataset_df = self._select_split(
                pd.concat(
                    [
                        train_df.assign(split=Split.TRAIN.value),
                        test_df.assign(split=Split.TEST.value),
                    ],
                    ignore_index=True,
                )
            )

    def _select_split(self, dataset_df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.split, list):
            split_values = {s.value if hasattr(s, "value") else s for s in self.split}
            selected = dataset_df[dataset_df["split"].isin(split_values)]
            return selected.reset_index(drop=True)

        split_value = self.split.value if hasattr(self.split, "value") else self.split
        return dataset_df[dataset_df["split"] == split_value].reset_index(drop=True)

    def _infer_mask_size(self):
        inferred_size = self._mask_size

        if self.wave_to_spectro is not None and len(self.dataset_df) > 0:
            sample_path = self.dataset_df.iloc[0]["file_path"]
            waveform, _ = self._load_waveform(sample_path)
            waveform = waveform.mean(dim=0, keepdim=True)

            if self.transform is not None:
                waveform = self.transform(waveform)

            with torch.no_grad():
                spectro = self.wave_to_spectro(waveform)

            inferred_size = tuple(int(dim) for dim in spectro.shape[-2:])
        elif self.img_size is not None:
            inferred_size = tuple(int(dim) for dim in self.img_size)

        self._mask_size = inferred_size
        self.dataset_arguments.gt_mask_size = inferred_size

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        file_path = self.dataset_df.iloc[idx]["file_path"]
        waveform, _ = self._load_waveform(file_path)

        # Keep a mono waveform as the model input.
        waveform = waveform.mean(dim=0, keepdim=False)

        if self.transform is not None:
            waveform = self.transform(waveform.unsqueeze(0)).squeeze(0)

        if self.split == Split.TRAIN:
            return waveform

        label = int(self.dataset_df.iloc[idx]["label"])
        mask = torch.zeros((1, *self._mask_size), dtype=torch.float32)
        return waveform, label, mask, file_path
