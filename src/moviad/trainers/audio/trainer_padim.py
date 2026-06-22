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

    def train(self, train_dataloader):
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
