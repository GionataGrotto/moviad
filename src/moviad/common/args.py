import torch
from dataclasses import dataclass

from moviad.datasets.builder import DatasetConfig, DatasetType
from moviad.utilities.configurations import Split

@dataclass
class Args:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    contamination_ratio: float = 0.0
    seed: int = 4
    dataset_config: DatasetConfig = None
    dataset_type: DatasetType = None
    batch_size: int = 4
    subset: float = 1.0
    streaming: bool = False
