from pathlib import Path
from typing import Union, Optional, NamedTuple, List
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import trange
import pdb


from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio import transforms

import soundata
from torch.utils.data import DataLoader, random_split


def trim_zeros(tensor: torch.Tensor) -> torch.Tensor:
    non_zero_indices = torch.nonzero(tensor, as_tuple=True)[0]
    if len(non_zero_indices) == 0:
        return tensor[:0]  # Return an empty tensor if no non-zeros
    return tensor[non_zero_indices[0] : non_zero_indices[-1] + 1]


def min_max_norm(x):
    d = x.max() - x.min()
    return (x - x.min()) / d if d != 0 else x


class SpectrogramBinarizer:

    @staticmethod
    def by_energy_threshold(threshold, spectrogram_dB, return_binary=True):
        """Find the threshold in dB energy that retains a certain percentage of the energy"""

        # check if the input is batched
        if len(spectrogram_dB.shape) == 4:
            spectrogram_dB = spectrogram_dB.squeeze(1)
            returns = []
            for s in spectrogram_dB:
                returns.append(
                    SpectrogramBinarizer.by_energy_threshold(
                        threshold, s, return_binary=return_binary
                    )
                )
            stacked = np.stack(returns)
            stacked = np.expand_dims(stacked, axis=1)
            return stacked

        log_spectro = spectrogram_dB.ravel()
        normalized_spectro = sorted(min_max_norm(log_spectro), reverse=True)
        # we need to min max normalize in order to have a cumulative sum that goes from 0 to 1
        cumulative = min_max_norm(np.cumsum(normalized_spectro))
        # at which index is the cumulative normalized dB energy above the threshold?
        threshold_idx = np.argmax(cumulative > threshold)
        # at that index, what is the (normalized) energy of the pixel?
        energy_threshold = normalized_spectro[threshold_idx]
        # create the binary mask based on the threshold
        binary_mask = min_max_norm(spectrogram_dB) > energy_threshold

        if return_binary:
            return binary_mask.astype(int)

        # create the thresholded spectrogram
        thresholded_spectro = spectrogram_dB.copy()
        # set all pixels below the threshold to the minimum value of the original spectrogram
        thresholded_spectro[~binary_mask] = np.min(spectrogram_dB)
        return cumulative, threshold_idx, thresholded_spectro, energy_threshold

    @staticmethod
    def by_quantile_threshold(threshold, spectrogram_dB, return_binary=True):
        log_spectro = spectrogram_dB.ravel()
        cumulative = np.cumsum(min_max_norm(log_spectro))
        # sort so that we can find the quantile threshold
        log_spectro = sorted(log_spectro, reverse=True)
        threshold_idx = int(threshold * len(log_spectro))
        energy_threshold = log_spectro[threshold_idx]

        binary_mask = spectrogram_dB > energy_threshold
        if return_binary:
            return binary_mask.astype(int)

        # create the thresholded spectrogram
        thresholded_spectro = spectrogram_dB.copy()
        # set all pixels below the threshold to the minimum value of the original spectrogram
        thresholded_spectro[~binary_mask] = np.min(spectrogram_dB)
        return cumulative, threshold_idx, thresholded_spectro, energy_threshold

    @staticmethod
    def _top_k_pooling(spectro, k=5):
        """
        Spectrogram shape: (n_frames, n_mels)
        """
        # for each frame, get the top k mel bins sum
        top_k = np.argsort(spectro, axis=1)[:, -k:]
        top_k_sum = np.take_along_axis(spectro, top_k, axis=1).sum(axis=1)
        # return shape: (n_frames,)
        return top_k_sum

    @staticmethod
    def temporal_top_k_threshold(threshold, spectrogram_dB, k=5):
        # check if the input is batched
        if len(spectrogram_dB.shape) == 4:
            spectrogram_dB = spectrogram_dB.squeeze(1)
            returns = []
            for s in spectrogram_dB:
                returns.append(
                    SpectrogramBinarizer.temporal_top_k_threshold(threshold, s)
                )
            stacked = np.stack(returns)
            stacked = np.expand_dims(stacked, axis=1)
            return stacked

        pooled_spectro = SpectrogramBinarizer._top_k_pooling(spectrogram_dB, k=k)
        binary_mask = (min_max_norm(pooled_spectro) > threshold).astype(int)
        return binary_mask


class TestSetSample(NamedTuple):
    """
    A sample from the test set.

    Attributes
    ----------
        audio_clip : torch.Tensor
            The audio clip which is the input to the model.
        label_clip_level : bool
            1 if the clip contains an anomaly, 0 otherwise.
        bg_clip : Optional[torch.Tensor]
            The background audio clip (if any).
        anomaly_clip : Optional[torch.Tensor]
            The anomaly audio clip (if any).
        anomaly_start_idx : Optional[int]
            The index at which the anomaly starts in the audio clip.
    """

    audio_clip: torch.Tensor
    label_clip_level: bool
    bg_clip: Optional[torch.Tensor]
    anomaly_clip: Optional[torch.Tensor]
    anomaly_start_idx: Optional[int]
    anomaly_end_idx: Optional[int]


class UrbanSound8KDataset(Dataset):

    CATEGORIES = [
        "air_conditioner",
        "car_horn",
        "children_playing",
        "dog_bark",
        "drilling",
        "engine_idling",
        "gun_shot",
        "jackhammer",
        "siren",
        "street_music",
    ]

    def __init__(
        self,
        path: Union[str, Path],
        target_duration: Optional[float] = None,
        categories_to_load: Optional[list] = None,
        download: bool = False,
        debug=False,
        target_sample_rate=44100,
    ):
        """
        Categories (or class labels) [str]: air_conditioner, car_horn, children_playing,
            dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music
        """

        path = Path(path) if isinstance(path, str) else path
        self.dataset = soundata.initialize("urbansound8k", data_home=path)
        if download:
            self.dataset.download()
        if debug:
            self.dataset.validate()

        self.clip_ids = self.dataset.clip_ids
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.is_training = False

        if target_duration is not None:
            self._filter_by_duration(target_duration)

        if categories_to_load is not None:
            self._filter_by_categories(categories_to_load)

    def train(self):
        self.is_training = True

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, idx):
        clip = self.dataset.clip(self.clip_ids[idx])
        waveform, sample_rate = torchaudio.load(clip.audio_path)

        # convert to mono
        waveform = waveform.mean(dim=0)

        if sample_rate != self.target_sample_rate:
            waveform = transforms.Resample(
                orig_freq=sample_rate, new_freq=self.target_sample_rate
            )(waveform)
            sample_rate = self.target_sample_rate

        # pad to the target duration
        if self.target_duration is not None:
            target_len = int(self.target_sample_rate * self.target_duration)
            waveform = torch.nn.functional.pad(
                waveform, (0, target_len - len(waveform)), "constant", 0
            )
            waveform = waveform[:target_len]

        return waveform, sample_rate  # , clip.tags

    def get_with_category(self, idx):
        clip = self.dataset.clip(self.clip_ids[idx])
        return *self[idx], clip.class_label

    def _filter_by_duration(self, min_duration):
        def duration(clip):
            return clip.freesound_end_time - clip.freesound_start_time

        self.clip_ids = [
            id
            for id in self.clip_ids
            if duration(self.dataset.clip(id)) >= min_duration
        ]

    def _filter_by_categories(self, categories_to_load):
        self.clip_ids = [
            id
            for id in self.clip_ids
            if self.dataset.clip(id).class_label in categories_to_load
        ]

    def _compute_durations(self):
        durations = []
        for i in trange(len(self), desc="Computing durations for filtering"):
            waveform, sample_rate = self[i]
            durations.append(len(waveform) / sample_rate)
        return np.array(durations)


class Esc50Dataset(Dataset):

    CATEGORIES = {
        "animals": [
            "cat",
            "cow",
            "crow",
            "dog",
            "frog",
            "hen",
            "insects",
            "pig",
            "rooster",
            "sheep",
        ],
        "natural_soundscapes": [
            "rain",
            "sea_waves",
            "crackling_fire",
            "crickets",
            "chirping_birds",
            "water_drops",
            "wind",
            "pouring_water",
            "thunderstorm",
        ],
        "human_non_speech": [
            "crying_baby",
            "sneezing",
            "clapping",
            "breathing",
            "coughing",
            "footsteps",
            "laughing",
            "brushing_teeth",
            "snoring",
            "drinking_sipping",
        ],
        "interior_domestic": [
            "door_wood_knock",
            "mouse_click",
            "keyboard_typing",
            "door_wood_creaks",
            "can_opening",
            "washing_machine",
            "vacuum_cleaner",
            "clock_alarm",
            "clock_tick",
            "glass_breaking",
        ],
        "exterior_urban": [
            "helicopter",
            "chainsaw",
            "siren",
            "car_horn",
            "engine",
            "train",
            "church_bells",
            "airplane",
            "fireworks",
            "hand_saw",
        ],
    }

    def __init__(
        self,
        path: Union[str, Path],
        max_memory_samples: Optional[int] = 128,
        categories_to_load: Optional[list] = None,
    ):
        """
        Parameters
        ----------
        path : Union[str, Path]
            Path to the ESC-50 dataset.
        max_memory_samples : Optional[int]
            Maximum number of audio clips to store in memory.
            If None, all audio clips are reloaded every time they are requested.
        """

        self.path = Path(path) if isinstance(path, str) else path

        # Load the metadata
        mpath = self.path / "meta" / "esc50.csv"
        assert mpath.exists(), f"Metadata file not found in {mpath}"
        self.data = pd.read_csv(mpath)
        assert all(
            self.data["filename"].map(lambda x: (self.path / "audio" / x).exists())
        ), "Audio files not found"

        # Audio files paths
        apath = self.path / "audio"
        self.data["path"] = self.data["filename"].map(lambda x: apath / x)

        # Filter categories
        if categories_to_load is not None:
            self.data = self.data[self.data["category"].isin(categories_to_load)]
            self.data.reset_index(drop=True, inplace=True)

        self.max_memory_samples = max_memory_samples
        self.audio = OrderedDict()  # only store the last max_memory_samples audio clips

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx in self.audio:
            return self.audio[idx]

        audio, sample_rate = torchaudio.load(str(self.data.loc[idx, "path"]))

        # convert to mono
        audio = audio.mean(dim=0)

        # save the audio clip in memory
        self.audio[idx] = audio, sample_rate
        if (self.max_memory_samples is not None) and (
            len(self.audio) > self.max_memory_samples
        ):
            self.audio.pop(next(iter(self.audio)))

        return audio, sample_rate

    def get_with_category(self, idx):
        return *self[idx], self.data.loc[idx, "category"]


class AudioAnomalyDataset(Dataset):

    def __init__(
        self,
        bg_data,
        contamination_data,
        SNR_dB_distribution=lambda: np.random.uniform(-6.0, 6.0),
        audio_start_distribution=lambda: np.random.uniform(0, 1),
        max_anomaly_len_distribution=lambda: 1.0,
        contamination=0.5,  # test set ratio between normal and anomalous audio clips
        clip_duration=4.0,
        random_state=None,
        target_sample_rate=44100,  # Hz
    ):
        from moviad.utilities.audio.audio_feature_exctractor import (
            AudioFeatureExtractor,
        )

        _, _, spectro_transform = AudioFeatureExtractor._load_spectrogram_transform(
            "Cnn14"
        )

        np.random.seed(random_state) if random_state else None

        self.bg_dataset = bg_data
        self.contamination_dataset = contamination_data
        self.SNR_dB_distribution = SNR_dB_distribution
        self.audio_start_distribution = audio_start_distribution
        self.max_anomaly_len_distribution = max_anomaly_len_distribution
        self.contamination = contamination
        self.clip_duration = clip_duration
        self.target_sample_rate = target_sample_rate
        self.img_shape = spectro_transform(
            torch.rand((1, int(target_sample_rate * clip_duration)))
        ).shape[-2:]

        # dict[int, str], mapping from dataset index to category of the anomaly used to create the mixture
        self.anomaly_category_mapping = dict()

        self.dataset = self._contaminate_data()

    @property
    def num_anomalies(self):
        return int(self.contamination * len(self.bg_dataset))

    @property
    def anomaly_classes_counts(self):
        return (
            pd.Series(list(self.anomaly_category_mapping.values()))
            .value_counts()
            .to_dict()
        )

    def _contaminate_data(self):

        anomalous_idxs = np.random.choice(
            len(self.bg_dataset), self.num_anomalies, replace=False
        )

        dataset:List[TestSetSample] = []

        # To use only with constant SNR
        snr_lin = 10 ** (self.SNR_dB_distribution() / 10)
        self.noise_factor = 1 / (snr_lin + 1)
        
        for i in trange(len(self.bg_dataset), desc="Contaminating dataset"):
            # load the background clip
            bg_clip, bg_sample_rate = self.bg_dataset[i]
            if bg_sample_rate != self.target_sample_rate:
                bg_clip = torchaudio.transforms.Resample(
                    orig_freq=bg_sample_rate, new_freq=self.target_sample_rate
                )(bg_clip)
            # pad / trim to the target duration
            target_len = int(self.target_sample_rate * self.clip_duration)
            bg_clip = torch.nn.functional.pad(
                bg_clip, (0, target_len - len(bg_clip)), "constant", 0
            )
            bg_clip = bg_clip[:target_len]
            # contaminate if needed
            if i in anomalous_idxs:
                anomaly_clip, start_idx, end_idx, clip = self._add_anomaly(bg_clip, i)
                label_clip_level = True
            else:
                anomaly_clip = None
                start_idx = None
                end_idx = None
                clip = bg_clip * (1 - self.noise_factor)
                label_clip_level = False

            # normalize as per speechbrain
            bg_clip *= 1 - self.noise_factor

            sample = TestSetSample(
                audio_clip=clip,
                label_clip_level=label_clip_level,
                bg_clip=bg_clip,
                anomaly_clip=anomaly_clip,
                anomaly_start_idx=start_idx,
                anomaly_end_idx=end_idx,
            )
            dataset.append(sample)
        
        self.global_max = torch.max(torch.stack([x.audio_clip for x in dataset]))

        for i, sample in enumerate(dataset):
            if sample.anomaly_clip is not None:
                anomaly_clip = sample.anomaly_clip / self.global_max
            else:
                anomaly_clip = None
            dataset[i] = TestSetSample(
                audio_clip=sample.audio_clip / self.global_max,
                label_clip_level=sample.label_clip_level,
                bg_clip=sample.bg_clip / self.global_max,
                anomaly_clip=anomaly_clip,
                anomaly_start_idx=sample.anomaly_start_idx,
                anomaly_end_idx=sample.anomaly_end_idx,
            )

        return dataset

    @staticmethod
    def _mix_signal_with_noise(signal, noise, snr_dB):
        """mixing like in [speechbrain](https://github.com/speechbrain/speechbrain/)"""
        dB_to_amplitude = lambda dB: 10 ** (dB / 20)

        # Assume signal and noise have the same length
        num_samples = torch.tensor(signal.size(0), dtype=torch.float32)

        # Compute RMS amplitude of the signal
        rms_signal = torch.norm(signal) / torch.sqrt(num_samples)

        # Compute noise scaling factor based on SNR
        noise_factor = 1 / (dB_to_amplitude(snr_dB) + 1)
        scaled_noise_amplitude = noise_factor * rms_signal

        # Rescale noise and add to the signal
        rms_noise = torch.norm(noise) / torch.sqrt(num_samples)
        noise *= scaled_noise_amplitude / (rms_noise + 1e-14)

        # Create the mixture
        mixture = signal * (1 - noise_factor) + noise

        # Normalize to prevent clipping
        mixture /= torch.max(torch.abs(mixture)).clamp(min=1.0)

        return mixture

    def _add_anomaly(self, bg_clip, dataset_idx: int):
        """
        Given a background clip, add an anomaly sound to it with a given SNR.
        """
        # load a randomly selected contamination clip
        contamination_idx = np.random.randint(0, len(self.contamination_dataset))
        anomaly_clip, anomaly_clip_sample_rate, category = (
            self.contamination_dataset.get_with_category(contamination_idx)
        )
        self.anomaly_category_mapping[dataset_idx] = category
        # resample if needed
        if anomaly_clip_sample_rate != self.target_sample_rate:
            anomaly_clip = torchaudio.transforms.Resample(
                orig_freq=anomaly_clip_sample_rate, new_freq=self.target_sample_rate
            )(anomaly_clip)

        # trim the anomaly sound to the max length
        anomaly_clip = trim_zeros(anomaly_clip)
        max_anomaly_len = int(len(bg_clip) * self.max_anomaly_len_distribution())
        if len(anomaly_clip) > max_anomaly_len:
            anomaly_clip = anomaly_clip[:max_anomaly_len]

        # pad the anomaly sound to match the background clip
        start = self.audio_start_distribution()
        usable_duration = len(bg_clip) - len(anomaly_clip)
        start_idx = int(start * usable_duration)
        end_idx = start_idx + len(anomaly_clip)
        padded_anomaly = torch.zeros_like(bg_clip)
        padded_anomaly[start_idx:end_idx] = anomaly_clip

        # sample the SNR
        snr = self.SNR_dB_distribution()

        # create the mixture
        mixture = self._mix_signal_with_noise(bg_clip, padded_anomaly, snr)

        # add anomaly to the clip
        return padded_anomaly, start_idx, end_idx, mixture

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class AudioAnomalyDatasetFF(AudioAnomalyDataset):

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        return sample.audio_clip, sample.label_clip_level, sample.bg_clip


class RetrofitTrainDs(Dataset):
    def __init__(self, dataset, global_max, noise_factor):
        self.dataset = dataset
        self.global_max = global_max
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate = self.dataset[idx]
        return (1 - self.noise_factor) * waveform / self.global_max


class RetrofitTestDs(Dataset):
    def __init__(self, dataset, wav_to_spectro, img_size):
        self.dataset = dataset
        self.wav_to_spectro = wav_to_spectro
        self.img_shape = (1, *img_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample: TestSetSample = self.dataset[idx]
        # model_input, labels, anomaly_masks, _
        if sample.anomaly_clip is not None:
            assert sample.anomaly_start_idx is not None
            anomaly_mask = self.wav_to_spectro(
                sample.anomaly_clip.unsqueeze(0)
            ).squeeze(0)
            anomaly_tmp_mask = torch.zeros(self.img_shape[1])
            anomaly_tmp_mask[
                sample.anomaly_start_idx
                // self.wav_to_spectro.hop_length : sample.anomaly_end_idx
                // self.wav_to_spectro.hop_length
            ] = 1
        else:
            anomaly_mask = torch.zeros(self.img_shape)
            anomaly_tmp_mask = torch.zeros(self.img_shape[1])

        return (
            sample.audio_clip,
            sample.label_clip_level,
            anomaly_mask,
            anomaly_tmp_mask,
            "",
        )


def generate_urban_esc_V1(
    urban_category,  # background clips category
    esc50_categories,
    wav_to_spectro,
    SNR_dB: float = 2,  # SNR for anomaly clips
    path_urban=Path.home() / "datasets" / "UrbanSound8K",  # background clips
    path_esc50=Path.home() / "datasets" / "ESC-50",  # anomaly clips
    seed=42,
    max_num_samples: Optional[int] = None,
    target_duration=4.0,
    target_sample_rate=44100,
):
    """
    anomalous_SNR: float, usually 0.5 (for -6dB), 1 (for 0dB), 2 (for 6dB)
    """

    background_ds = UrbanSound8KDataset(
        path_urban,
        download=not path_urban.exists(),
        target_duration=target_duration,
        categories_to_load=[urban_category],
        target_sample_rate=target_sample_rate,
    )

    if max_num_samples is not None:
        background_ds, _ = random_split(
            background_ds,
            [max_num_samples, len(background_ds) - max_num_samples],
            generator=torch.Generator().manual_seed(seed),
        )

    anomaly_ds = Esc50Dataset(path_esc50, categories_to_load=esc50_categories)

    train_size = int(0.75 * len(background_ds))
    test_size = len(background_ds) - train_size

    train_dataset, test_bg_ds = random_split(
        background_ds,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    test_dataset = AudioAnomalyDataset(
        test_bg_ds,
        anomaly_ds,
        random_state=seed,
        SNR_dB_distribution=lambda: SNR_dB,
    )  # contamination

    test_dataset_ff = AudioAnomalyDatasetFF(
        test_bg_ds,
        anomaly_ds,
        random_state=seed,
        SNR_dB_distribution=lambda: SNR_dB,
    )  # contamination

    train_dataset = RetrofitTrainDs(train_dataset, test_dataset.global_max, test_dataset.noise_factor)
    test_dataset_retro = RetrofitTestDs(
        test_dataset, wav_to_spectro, img_size=test_dataset.img_shape
    )

    return train_dataset, test_dataset_retro, test_dataset_ff
