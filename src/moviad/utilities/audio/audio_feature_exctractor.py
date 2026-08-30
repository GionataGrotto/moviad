"""
This class represent an audio feature extractor built on top of a custom backbone. The backbone name is passed as input to the constructor.
The features are extracted from the audio spectograms.
Based on the model name and the layers indexes it will consider the correct layers for the feature extraction.
"""

from __future__ import annotations
from pathlib import Path
import copy

import torch
import torchvision
from torchlibrosa.stft import Spectrogram
from torchlibrosa.stft import LogmelFilterBank

from moviad.backbones.clap.clap import AudioEncoder, Cnn14

SUPPORTED_BACKBONES = ["Cnn14", "Cnn14_finetuned"]


class AudioFeatureExtractor:

    def __init__(
        self,
        model_name: str,
        layers_idx: list,
        device: torch.device,
        frozen: bool = True,
        pre_trained: bool = True,
        enable_spectrogram_transform: bool = True
    ):
        """ 
        Constructor

        Args:
            model_name (str): name of the backbone to use
            layers_idx (list): list of layers identifiers
            device (torch.device): device to be used
        """

        self.model_name = model_name
        self.layers_idx = layers_idx
        self.device = device
        self.spectrogram_transform_enabled = enable_spectrogram_transform

        # check for backbone support
        if model_name not in SUPPORTED_BACKBONES:
            raise Exception(
                f"The backbone: {model_name} is not yet supported for feature extraction"
            )

        # load the model
        if model_name == "Cnn14":
            self._load_cnn14(pre_trained)
        elif model_name == "Cnn14_finetuned":
            self._load_cnn14_finetuned()

        # attach hooks
        self.attach_hook()

        # load the model to the device
        self.model = self.model.to(self.device)
        self.spectrogram_extractor.to(self.device)
        self.logmel_extractor.to(self.device)

        self.spectrogram_extractor.eval()
        for parameter in self.spectrogram_extractor.parameters():
            parameter.requires_grad = False
        
        self.logmel_extractor.eval()
        for parameter in self.logmel_extractor.parameters():
            parameter.requires_grad = False

        # freeze the model if needed
        if frozen:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def to(self, device: torch.device | str):
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.spectrogram_extractor = self.spectrogram_extractor.to(self.device)
        self.logmel_extractor = self.logmel_extractor.to(self.device)
        return self

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    @staticmethod
    def _load_spectrogram_transform(model_name):
        """
        Load the spectrogram and mel spectrogram extraction

        Args:
            model_name (str): name of the backbone used for feature extraction

        Returns:
            tuple: spectrogram_extractor, logmel_extractor, spectro_transform
        """
        
        if "Cnn14" in model_name:

            # set spectrogram and mel spectrogram extraction
            n_fft = 1024
            hop_length = 320
            win_length = 1024

            spectrogram_extractor = Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window="hann",
                center=True,
                pad_mode="reflect",
                freeze_parameters=True,
            )

            sample_rate = 44100
            win_length = 1024
            n_mels = 64
            fmin = 50
            fmax = 14000

            logmel_extractor = LogmelFilterBank(
                sr=sample_rate,
                n_fft=win_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
                ref=1.0,
                amin=1e-10,
                top_db=None,
                freeze_parameters=True,
            )
            
            spectro_transform = torch.nn.Sequential(
                copy.deepcopy(spectrogram_extractor), 
                copy.deepcopy(logmel_extractor)
            )

            spectro_transform.hop_length = hop_length
            spectro_transform.win_length = win_length
            
            return spectrogram_extractor, logmel_extractor, spectro_transform
        else:
            raise NotImplementedError(f"Model {model_name} not supported")

    def _load_cnn14(self, pretrained=True):

        self.spectrogram_extractor, self.logmel_extractor, self.spectro_transform = (
            self._load_spectrogram_transform(self.model_name)
        )

        out_emb = 2048
        d_proj = 1024
        classes_num = 527

        self.model = AudioEncoder(
            self.model_name,
            out_emb,
            d_proj,
            classes_num,
        )

        if pretrained:
            p = Path(__file__).resolve().parents[2] / "weights" / "clap_encoder.pth"
            assert p.exists(), f"AudioFeatureExtractor Cnn14 weights not found in path: {p}"
            self.model.load_state_dict(
                torch.load(p, map_location=self.device, weights_only=False)
            )

    def _load_cnn14_finetuned(self):

        self.spectrogram_extractor, self.logmel_extractor, self.spectro_transform = (
            self._load_spectrogram_transform(self.model_name)
        )

        out_emb = 2048
        d_proj = 1024
        classes_num = 527

        self.model = Cnn14(1, out_emb)

        p = Path(__file__).resolve().parents[2] / "weights" / "finetuned_cnn14.pth"
        assert p.exists(), f"AudioFeatureExtractor Cnn14 weights not found in path: {p}"
        self.model.load_state_dict(torch.load(p))

    def attach_hook(self, bootstrap_idx=0):

        def hook(module, input, output):
            self.features.append(output)

        if self.model_name == "Cnn14":

            for idx in self.layers_idx:
                # example layers: ["conv_block2", "conv_block3", "conv_block4"]
                getattr(self.model.base, idx).register_forward_hook(hook)

        elif self.model_name == "Cnn14_finetuned":

            for idx in self.layers_idx:
                # example layers: ["conv_block2", "conv_block3", "conv_block4"]
                getattr(self.model, idx).register_forward_hook(hook)

    def wavs_to_spectros(self, batch: torch.Tensor) -> torch.Tensor:

        batch = batch.to(self.device)
        batch = self.spectrogram_extractor(batch)
        batch = self.logmel_extractor(batch)

        return batch

    def __call__(self, batch: torch.Tensor) -> list[torch.Tensor]:

        if self.spectrogram_transform_enabled:
            batch = self.wavs_to_spectros(batch)

        self.spec_shape = batch.shape

        self.features = []

        self.model(batch)
        
        return self.features
    
    def disable_wavs_to_spectros(self):
        self.spectrogram_transform_enabled = False
