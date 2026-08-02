import wandb
import torch

from tqdm import tqdm
import os
from typing import Union
from torch.utils.data import DataLoader

from moviad.models.audio.patchcore.patchcore import PatchCore
from moviad.models.audio.patchcore.kcenter_greedy import KCenterGreedy
from moviad.utilities.audio.evaluator_audio import AudioEvaluator


class TrainerPatchCore:
    """
    This class contains the code for training the CFA model

    Args:
        patchore_model (PatchCore): model to be trained
        train_dataloder (torch.utils.data.DataLoader): train dataloader
        test_dataloder (torch.utils.data.DataLoader): test dataloader
        device (str): device to be used for the training
    """

    def __init__(
        self,
        patchore_model: PatchCore,
        train_dataloader: DataLoader,
        test_dataloder: DataLoader,
        device: Union[str, torch.device],
        force_cpu:bool = False
    ):
        self.patchore_model = patchore_model
        self.train_dataloader = train_dataloader
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.force_cpu = force_cpu

    def train(self):
        """
        This method trains the PatchCore model and evaluate it at the end of training
        """

        embeddings = []

        with torch.no_grad():
            print("Embedding Extraction:")
            for batch in tqdm(iter(self.train_dataloader)):
                if isinstance(batch, tuple):
                    embedding = self.patchore_model(batch[0].to(self.device))
                else:
                    embedding = self.patchore_model(batch.to(self.device))

                embeddings.append(embedding.cpu())

            embeddings = torch.cat(embeddings, dim=0)
            torch.cuda.empty_cache()

            print("Coreset Extraction:")
            sampler = KCenterGreedy(embeddings, self.device)
            sampled_idxs = sampler.get_coreset_idx_randomp(
                embeddings,
                memory_bank_size=self.patchore_model.memory_bank_size,
                force_cpu=self.force_cpu,
            )
            self.patchore_model.memory_bank = embeddings[sampled_idxs]