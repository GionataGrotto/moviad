import glob
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import torch

from moviad.utilities.configurations import LabelName, Split

class MIMIDataset(Dataset):

    def __init__(
        self,
        dataset_directory:str,
        snr:str,
        category:str,
        machine_id: str,
        split:Split,
        wave_to_spectro=None, 
        transform=None,
    ):
        """
        Args:
            dataset_directory (str): Path to the dataset directory.
            snr (str): Signal to noise ratio.
            category (str): Category of the dataset.
            machine_id (str): id of the machine.
            split (Split): Split of the dataset.
            wave_to_spectro: spectrogram extractor from the wav. 
            transform (callable, optional): Optional transform to be applied on a sample.

        NB: the wave_to_spectro is used only for getting the shape of the spectrogram of a sample 
        in test mode.
        """

        self.dataset_directory = dataset_directory
        self.snr = snr
        self.category = category
        self.split = split
        self.wave_to_spectro = wave_to_spectro
        self.transform = transform
        self.machine_id = machine_id
        self._load_dataset()

    def _load_dataset(self):

        load_directory = f"{self.dataset_directory}/{self.snr}/{self.category}/{self.machine_id}"

        # Get the list of all files in the dataset directory
        files = [f for f in glob.glob(load_directory + "/**/*.wav", recursive=True)]

        dataset_df = pd.DataFrame(files, columns=["file_path"])
        dataset_df["label"] = dataset_df["file_path"].map(lambda file_path: Path(file_path).parent.name)
        dataset_df["label"] = dataset_df.apply( lambda row: LabelName.NORMAL if row["label"] == "normal" else LabelName.ABNORMAL, axis=1)

        # create train and test splits
        normal_samples = dataset_df[dataset_df["label"] == LabelName.NORMAL]
        abnormal_samples = dataset_df[dataset_df["label"] == LabelName.ABNORMAL]

        test_normal_samples = normal_samples.sample(n=int(len(abnormal_samples)*0.3))
        test_abnormal_samples = abnormal_samples.sample(n=int(len(abnormal_samples)*0.3))
        test_df = pd.concat([test_normal_samples, test_abnormal_samples])

        train_df = dataset_df.drop(test_df.index)

        # Reset the index for both dataframes
        test_df = test_df.reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)

        if self.split == Split.TRAIN:
            self.dataset_df = train_df
        else:
            test_df = test_df.sample(frac=1)
            self.dataset_df = test_df

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the dataset.

        Returns:
            waveform (Tensor): Audio waveform, if the  split is TRAIN.
            waveform, label, gt_mask, path (Tensor, int): 
                Audio waveform, label, ground_truth_mask and sample if the split is TEST.
            
        NB: the gt_mask is a tensor of zeros with the same shape of the spectrogram since in the
        mimii dataset only sample level labels are available.
        """

        file_path = self.dataset_df.iloc[idx, 0]

        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)

        if self.transform:
            waveform = self.transform(waveform)

        # mean the channels
        waveform = waveform.mean(dim=0, keepdim=False)

        if self.split == Split.TRAIN:
            return waveform
        else:
            gt_masks = torch.zeros(1, 1379, 64)
            anomaly_tmp_mask = torch.zeros(1379)
            # waveform, label, gt_mask, path
            return waveform, self.dataset_df.iloc[idx, 1], gt_masks, anomaly_tmp_mask, file_path
