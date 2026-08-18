from __future__ import annotations
from tqdm import tqdm
import torch

from moviad.models.padim.padim import Padim
from moviad.trainers.trainer import Trainer, TrainerResult


class TrainerPadim(Trainer):

    def __init__(
        self,
        model: Padim,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader | None,
        device,
        apply_diagonalization=False,
        logger=None,
    ):
        """
        Args:
            device: one of the following strings: 'cpu', 'cuda', 'cuda:0', ...
        """
        super().__init__(model, train_dataloader, eval_dataloader, device, logger)
        self.apply_diagonalization = apply_diagonalization

    def train(self, streaming: bool = False):
        print(f"Train Padim. Backbone: {self.model.backbone_model_name}")

        self.model.train()

        if self.logger is not None:
            self.logger.watch(self.model)

        if streaming:
            self._fit_streaming()
            metrics = self.evaluator.evaluate(self.model)
            if self.logger is not None:
                self.logger.log(metrics)
            return TrainerResult(**metrics)

        # 1. get the feature maps from the backbone
        layer_outputs: dict[str, list[torch.Tensor]] = {
            layer: [] for layer in self.model.layers_idxs
        }
        for x in tqdm(self.train_dataloader, "| feature extraction | train | %s |"):
            outputs = self.model(x.to(self.device))
            assert isinstance(outputs, dict)
            for layer, output in outputs.items():
                layer_outputs[layer].extend(output)

        # 2. use the feature maps to get the embeddings
        embedding_vectors = self.model.raw_feature_maps_to_embeddings(layer_outputs)

        
        

        # 3. fit the multivariate Gaussian distribution
        if self.apply_diagonalization:
            self.model.fit_multivariate_diagonal_gaussian(
                embedding_vectors, update_params=True, logger=self.logger
            )
        else:
            self.model.fit_multivariate_gaussian(
                embedding_vectors, update_params=True, logger=self.logger
            )

        metrics = self.evaluator.evaluate(self.model)

        if self.logger is not None:
            self.logger.log(metrics)

        print("End training performances:")
        self.print_metrics(metrics)

        return TrainerResult(**metrics)

    def _fit_streaming(self):
        count = 0
        mean = None
        m2 = None
        for x in tqdm(self.train_dataloader, "| streaming feature extraction | train |"):
            outputs = self.model(x.to(self.device))
            layer_outputs = {layer: [output] for layer, output in outputs.items()}
            embeddings = self.model.raw_feature_maps_to_embeddings(layer_outputs)
            batch, channels, height, width = embeddings.shape
            values = embeddings.detach().cpu().double().permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            batch_count = values.shape[0]
            batch_mean = values.mean(dim=0)
            centered = values - batch_mean
            batch_m2 = torch.einsum("npc,npd->pcd", centered, centered)
            if mean is None:
                mean, m2, count = batch_mean, batch_m2, batch_count
                continue
            total = count + batch_count
            delta = batch_mean - mean
            m2 = m2 + batch_m2 + delta.unsqueeze(-1) * delta.unsqueeze(-2) * (count * batch_count / total)
            mean = mean + delta * (batch_count / total)
            count = total
        if count == 0:
            raise RuntimeError("Cannot stream-train PaDiM on an empty dataset")
        covariance = m2 / max(count - 1, 1)
        covariance = covariance + 0.01 * torch.eye(covariance.shape[-1], dtype=covariance.dtype).unsqueeze(0)
        if self.model.diag_cov:
            covariance = torch.diag_embed(torch.diagonal(covariance, dim1=-2, dim2=-1))
        self.model.gauss_mean = mean.float().transpose(0, 1).reshape(channels, height, width).numpy()
        self.model.gauss_cov = covariance.float().permute(2, 1, 0).reshape(channels, channels, height * width).numpy()
