import os
from tqdm import tqdm
import torch

from moviad.models.audio.padim.padim import Padim


class PadimTrainer:

    def __init__(self, model: Padim, device, save_path, data_path, class_name):
        """
        Args:
            device: one of the following strings: 'cpu', 'cuda', 'cuda:0', ...
        """
        self.model = model
        self.save_path = save_path
        self.class_name = class_name
        self.device = device

        model.to(device)

    def train(self, train_dataloader, streaming: bool = False):
        if streaming:
            return self.train_streaming(train_dataloader)

        print(f"Train Padim. Backbone: {self.model.backbone_model_name}")


        self.model.train()

        # 1. get the feature maps from the backbone
        layer_outputs: dict[str, list[torch.Tensor]] = {
            layer: [] for layer in self.model.layers_idxs
        }
        for x in tqdm(
            train_dataloader, "| feature extraction | train | %s |" % self.class_name
        ):
            outputs = self.model(x.to(self.device))
            assert isinstance(outputs, dict)
            for layer, output in outputs.items():
                layer_outputs[layer].extend(output)

        # 2. use the feature maps to get the embeddings
        embedding_vectors = self.model.raw_feature_maps_to_embeddings(layer_outputs)
        # 3. fit the multivariate Gaussian distribution
        self.model.fit_multivariate_gaussian(embedding_vectors, update_params=True)
        # 4. save the model
        if self.save_path is not None:
            model_savepath = self.model.get_model_savepath(self.save_path)
            os.makedirs(os.path.dirname(model_savepath), exist_ok=True)
            torch.save(self.model.state_dict(), model_savepath)

    def train_streaming(self, train_dataloader):
        """Fit PaDiM with online mean/covariance updates per spatial location."""
        count = 0
        mean = None
        m2 = None

        self.model.train()
        for x in tqdm(train_dataloader, "| streaming feature extraction | train | %s |" % self.class_name):
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
