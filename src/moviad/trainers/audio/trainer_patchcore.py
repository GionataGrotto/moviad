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

    def train(self, streaming: bool = False):
        """
        This method trains the PatchCore model and evaluate it at the end of training
        """

        if streaming:
            return self.train_streaming()

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

    def train_streaming(self):
        """Fit PatchCore with bounded memory using reservoir sampling.

        The feature extractor is still evaluated batch by batch, but all
        embeddings are discarded after the batch. The resulting memory bank is
        an unbiased sample of at most ``memory_bank_size`` embeddings.
        """
        reservoir = None
        seen = 0
        generator = torch.Generator(device="cpu").manual_seed(0)

        with torch.no_grad():
            print("Streaming embedding extraction:")
            for batch in tqdm(iter(self.train_dataloader)):
                inputs = batch[0] if isinstance(batch, tuple) else batch
                embeddings = self.patchore_model(inputs.to(self.device)).detach().cpu()
                for embedding in embeddings:
                    seen += 1
                    if reservoir is None:
                        reservoir = torch.empty(
                            (self.patchore_model.memory_bank_size, embedding.numel()),
                            dtype=embedding.dtype,
                        )
                    if seen <= reservoir.shape[0]:
                        reservoir[seen - 1].copy_(embedding.reshape(-1))
                    else:
                        replacement = int(torch.randint(seen, (1,), generator=generator))
                        if replacement < reservoir.shape[0]:
                            reservoir[replacement].copy_(embedding.reshape(-1))

        if reservoir is None:
            self.patchore_model.memory_bank = torch.empty(0)
        else:
            self.patchore_model.memory_bank = reservoir[: min(seen, reservoir.shape[0])]
