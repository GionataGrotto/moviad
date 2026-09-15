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
    for prefix in ("module.", "model."):
        if all(key.startswith(prefix) for key in normalized):
            normalized = {
                key.removeprefix(prefix): value for key, value in normalized.items()
            }

    # Official/full CLAP checkpoints include text encoders and CLAP projection
    # layers in addition to the audio network.  AudioEncoder only needs the
    # Cnn14 branch, which is commonly named ``audio_branch`` in those files.
    for prefix in ("audio_branch.", "audio_encoder."):
        audio_weights = {
            key.removeprefix(prefix): value
            for key, value in normalized.items()
            if key.startswith(prefix)
        }
        if audio_weights:
            normalized = audio_weights
            break
    return normalized


class AudioFeatureExtractor(torch.nn.Module):

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

        super().__init__()
        self.frozen = frozen
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
        super().to(self.device)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.train(mode and not self.frozen)
        self.spectrogram_extractor.eval()
        self.logmel_extractor.eval()
        self.spectro_transform.eval()
        return self

    def eval(self):
        return self.train(False)

    @staticmethod
    def sample_rate_for(model_name: str) -> int:
        if model_name not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported audio backbone: {model_name}")
        return 48000 if model_name == "HTSAT-base" else 44100

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
                freq_ratio = self.model.freq_ratio

                def htsat_hook(module, input, output, freq_ratio=freq_ratio):
                    tokens = output[0] if isinstance(output, tuple) else output
                    # Every HTSAT stage keeps a square token grid (the model
                    # squares the spectrogram up front, see reshape_wav2img),
                    # but the side shrinks at each of the first num_layers - 1
                    # stages (PatchMerging) and then stays fixed at the last
                    # one. Deriving it from the token count instead of a fixed
                    # per-stage formula keeps this correct for every stage.
                    side = round(tokens.shape[1] ** 0.5)
                    grid = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
                    self.features.append(
                        AudioFeatureExtractor._unshuffle_htsat_tokens(grid, freq_ratio)
                    )

                layer.register_forward_hook(htsat_hook)

        elif self.model_name == "Cnn14_finetuned":

            for idx in self.layers_idx:
                # example layers: ["conv_block2", "conv_block3", "conv_block4"]
                getattr(self.model, idx).register_forward_hook(hook)

    @staticmethod
    def _unshuffle_htsat_tokens(tokens_grid: torch.Tensor, freq_ratio: int) -> torch.Tensor:
        """Undo HTSAT's "reshape_wav2img" trick on a hooked token grid.

        HTSAT squares a long, narrow (time, frequency) spectrogram by slicing
        the time axis into ``freq_ratio`` contiguous chunks and stacking them
        along the frequency axis, so its Swin backbone sees a roughly square
        image. That makes the intermediate token grid's height axis a mix of
        (chunk index, true frequency) and its width axis "time within one
        chunk" -- not a plain (time, frequency) map. Feeding that raw grid to
        code that expects (time, frequency), such as ``F.interpolate`` to the
        spectrogram size or the top-k-over-frequency temporal score, silently
        scrambles both the pixel-level anomaly map and the per-time-frame
        score. This reproduces the un-scrambling HTSAT's own final
        ``forward_features`` head performs, generalised to any intermediate
        stage's resolution, restoring a genuine (frequency, time) layout with
        every chunk back in chronological order.

        Args:
            tokens_grid: hooked stage output reshaped to ``[B, C, side, side]``.
            freq_ratio: ``HTSAT_Swin_Transformer.freq_ratio`` of the model
                (number of chunks the time axis was split into).

        Returns:
            Tensor of shape ``[B, C, side // freq_ratio, side * freq_ratio]``,
            genuinely ordered as (frequency, time).
        """
        batch, channels, side, _ = tokens_grid.shape
        freq_sub = side // freq_ratio
        x = tokens_grid.reshape(batch, channels, freq_ratio, freq_sub, side)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        return x.reshape(batch, channels, freq_sub, freq_ratio * side)

    def wavs_to_spectros(self, batch: torch.Tensor) -> torch.Tensor:

        batch = batch.to(self.device)
        batch = self.spectrogram_extractor(batch)
        batch = self.logmel_extractor(batch)

        return batch

    def _forward_htsat_single_window(self, spectrogram: torch.Tensor) -> list[torch.Tensor]:
        """Run one HTSAT-base forward pass on a spectrogram within its fixed window."""
        self.features = []
        x = spectrogram.transpose(1, 3)
        x = self.model.bn0(x).transpose(1, 3)
        x = self.model.reshape_wav2img(x)
        self.model.forward_features(x)
        return self.features

    def _forward_htsat_windowed(self, spectrogram: torch.Tensor) -> list[torch.Tensor]:
        """Run HTSAT-base on a spectrogram longer than its fixed input window.

        HTSAT's Swin transformer only accepts clips up to
        ``spec_size * freq_ratio`` time frames (about 10.24s with this
        project's 48kHz/hop-480 frontend) because its positional embeddings
        and window attention are tied to a fixed resolution. Longer clips are
        split into non-overlapping windows of at most that many frames (the
        last one zero-padded, so real content is never time-stretched to fill
        the window the way ``reshape_wav2img`` would), each window is run
        through the normal single-window path, and the per-layer feature maps
        are concatenated back along the true time axis -- the last axis after
        ``_unshuffle_htsat_tokens`` -- with the padded tail of the last
        window's contribution cropped out proportionally to how much of it
        was real audio.

        Assumes every clip in the batch has the same length, which holds for
        every dataset in this benchmark (a batch is collated from
        fixed-duration clips).
        """
        target_T = self.model.spec_size * self.model.freq_ratio
        total_T = spectrogram.shape[2]

        window_maps: list[list[torch.Tensor]] | None = None
        start = 0
        while start < total_T:
            real_len = min(target_T, total_T - start)
            window = spectrogram[:, :, start:start + real_len, :]
            if real_len < target_T:
                pad = spectrogram.new_zeros(
                    spectrogram.shape[0], spectrogram.shape[1],
                    target_T - real_len, spectrogram.shape[3],
                )
                window = torch.cat((window, pad), dim=2)

            window_features = self._forward_htsat_single_window(window)
            if window_maps is None:
                window_maps = [[] for _ in window_features]

            for layer_idx, feature_map in enumerate(window_features):
                output_T = feature_map.shape[-1]
                valid_T = max(1, round(output_T * real_len / target_T))
                window_maps[layer_idx].append(feature_map[..., :valid_T])

            start += target_T

        return [torch.cat(maps, dim=-1) for maps in window_maps]

    def forward(self, batch: torch.Tensor) -> list[torch.Tensor]:

        if self.spectrogram_transform_enabled:
            batch = self.wavs_to_spectros(batch)

        self.spec_shape = batch.shape

        if self.model_name == "HTSAT-base":
            target_T = self.model.spec_size * self.model.freq_ratio
            if batch.shape[2] > target_T:
                self.features = self._forward_htsat_windowed(batch)
            else:
                self.features = self._forward_htsat_single_window(batch)
        else:
            self.features = []
            self.model(batch)

        return self.features

    def disable_wavs_to_spectros(self):
        self.spectrogram_transform_enabled = False
