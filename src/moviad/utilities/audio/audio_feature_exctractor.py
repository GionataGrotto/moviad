"""
This class represent an audio feature extractor built on top of a custom backbone. The backbone name is passed as input to the constructor.
The features are extracted from the audio spectograms.
Based on the model name and the layers indexes it will consider the correct layers for the feature extraction.
"""

from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
import copy

import torch
import torchvision
from torchlibrosa.stft import Spectrogram
from torchlibrosa.stft import LogmelFilterBank

from moviad.backbones.clap.clap import AudioEncoder, Cnn14, build_htsat_base

SUPPORTED_BACKBONES = ["Cnn14", "Cnn14_finetuned", "HTSAT-base"]


def _checkpoint_state_dict(checkpoint) -> dict[str, torch.Tensor]:
    """Return encoder weights from a raw or training-style PyTorch checkpoint.

    Historical Cnn14 weights in this project are stored directly as an
    ``AudioEncoder.state_dict()``.  Other training scripts save metadata such
    as the epoch and optimizer alongside the actual weights under
    ``state_dict``.  Both forms must load identically in benchmark runners.
    """
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Expected a checkpoint mapping or state_dict, got "
            f"{type(checkpoint).__name__}"
        )

    state = checkpoint
    for key in ("state_dict", "model_state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            break

    if not state or not all(isinstance(key, str) for key in state):
        raise ValueError("Checkpoint does not contain a valid string-keyed state_dict")

    # DataParallel and several training wrappers prepend these prefixes.  Only
    # remove a prefix when every key has it, preserving already-compatible
    # state_dicts such as the existing ``clap_encoder.pth``.
    normalized = dict(state)
    for prefix in ("module.", "model.", "audio_encoder."):
        if all(key.startswith(prefix) for key in normalized):
            normalized = {
                key.removeprefix(prefix): value for key, value in normalized.items()
            }
    return normalized


class AudioFeatureExtractor:

    def __init__(
        self,
        model_name: str,
        layers_idx: list,
        device: torch.device,
        frozen: bool = True,
        pre_trained: bool = True,
        enable_spectrogram_transform: bool = True,
        checkpoint_path: str | Path | None = None,
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
        self.checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None

        # check for backbone support
        if model_name not in SUPPORTED_BACKBONES:
            raise Exception(
                f"The backbone: {model_name} is not yet supported for feature extraction"
            )

        # load the model
        if model_name == "Cnn14":
            self._load_cnn14(pre_trained, self.checkpoint_path)
        elif model_name == "Cnn14_finetuned":
            self._load_cnn14_finetuned()
        elif model_name == "HTSAT-base":
            self._load_htsat_base(pre_trained, self.checkpoint_path)

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
        elif model_name == "HTSAT-base":
            spectrogram_extractor = Spectrogram(
                n_fft=1024, hop_length=480, win_length=1024,
                window="hann", center=True, pad_mode="reflect",
                freeze_parameters=True,
            )
            logmel_extractor = LogmelFilterBank(
                sr=48000, n_fft=1024, n_mels=64, fmin=50, fmax=14000,
                ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True,
            )
            spectro_transform = torch.nn.Sequential(
                copy.deepcopy(spectrogram_extractor),
                copy.deepcopy(logmel_extractor),
            )
            spectro_transform.hop_length = 480
            spectro_transform.win_length = 1024
            return spectrogram_extractor, logmel_extractor, spectro_transform
        else:
            raise NotImplementedError(f"Model {model_name} not supported")

    def _load_cnn14(self, pretrained=True, checkpoint_path: Path | None = None):

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
            p = checkpoint_path or (Path(__file__).resolve().parents[2] / "weights" / "clap_encoder.pth")
            assert p.exists(), f"AudioFeatureExtractor Cnn14 weights not found in path: {p}"
            checkpoint = torch.load(p, map_location=self.device, weights_only=False)
            self.model.load_state_dict(_checkpoint_state_dict(checkpoint))

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

    def _load_htsat_base(self, pretrained=True, checkpoint_path: Path | None = None):
        self.spectrogram_extractor, self.logmel_extractor, self.spectro_transform = (
            self._load_spectrogram_transform("HTSAT-base")
        )
        self.model = build_htsat_base()
        if pretrained:
            p = checkpoint_path or (Path(__file__).resolve().parents[2] / "weights" / "music_speech_audioset_epoch_15_esc_89.98.pt")
            if not p.exists():
                raise FileNotFoundError(f"HTSAT-base weights not found at {p}")
            checkpoint = torch.load(p, map_location="cpu", weights_only=False)
            state = checkpoint.get("state_dict", checkpoint)
            audio_state = {
                key.removeprefix("module.audio_branch."): value
                for key, value in state.items()
                if key.startswith("module.audio_branch.")
            }
            missing, unexpected = self.model.load_state_dict(audio_state, strict=False)
            if unexpected or any(not key.startswith(("head", "tscam_conv")) for key in missing):
                raise RuntimeError(
                    f"Invalid HTSAT-base checkpoint: missing={missing[:5]}, "
                    f"unexpected={unexpected[:5]}"
                )

    def attach_hook(self, bootstrap_idx=0):

        def hook(module, input, output):
            self.features.append(output)

        if self.model_name == "Cnn14":

            for idx in self.layers_idx:
                # example layers: ["conv_block2", "conv_block3", "conv_block4"]
                getattr(self.model.base, idx).register_forward_hook(hook)

        elif self.model_name == "HTSAT-base":
            for idx in self.layers_idx:
                layer_idx = int(idx) if str(idx).isdigit() else int(str(idx).split(".")[-1])
                layer = self.model.layers[layer_idx]

                def htsat_hook(module, input, output, layer_idx=layer_idx):
                    tokens = output[0] if isinstance(output, tuple) else output
                    # HTSAT has 64x64 patch tokens, halved at every stage.
                    side = 64 // (2 ** layer_idx)
                    self.features.append(
                        tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
                    )

                layer.register_forward_hook(htsat_hook)

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

        if self.model_name == "HTSAT-base":
            x = batch.transpose(1, 3)
            x = self.model.bn0(x).transpose(1, 3)
            x = self.model.reshape_wav2img(x)
            self.model.forward_features(x)
        else:
            self.model(batch)
        
        return self.features
    
    def disable_wavs_to_spectros(self):
        self.spectrogram_transform_enabled = False
