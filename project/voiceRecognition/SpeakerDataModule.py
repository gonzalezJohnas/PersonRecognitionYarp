from torch.utils.data import Dataset, DataLoader, random_split
import os, glob
import pandas as pd
import pytorch_lightning as pl
import torchaudio, torch
import numpy as np
from project.voiceRecognition.utils import save_audio, split_audio_chunks, process_dataset
import torchaudio.transforms as T


class SpeakerDataset(Dataset):
    """Speaker raw audio dataset."""

    def __init__(self, root_dir, transform=None, train=False):
        """
        Args:
            root_dir (string): Directory with all the subdirectory for each speaker and their audio.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if train:
            self.root_dir = os.path.join(root_dir, "train")
        else:
            self.root_dir = os.path.join(root_dir, "test")
        self.speaker_frame, self.name_dict = self._create_speaker_dataframe()
        self.transform = transform
        self.sample_rate = 16000
        self.resample_trans = torchaudio.transforms.Resample(48000, self.sample_rate)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=80, iid_masks=True)
        self.time_masking = T.TimeMasking(time_mask_param=80, iid_masks=True)

    def __len__(self):
        return len(self.speaker_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root_dir,
                                  self.speaker_frame.iloc[idx, 1])
        waveform, sample_rate = torchaudio.load(audio_name)
        waveform = self.resample_trans(waveform)

        label = self.speaker_frame.iloc[idx, 0]
        label = np.array([label])
        label = label.astype('long')
        sample = {'mfcc': waveform, 'label': label}

        sample["mfcc"] = self.transform(sample["mfcc"])
        rand = np.random.randint(0, 10)
        if 4 < rand <= 7:
            sample["mfcc"] = self.freq_masking(sample["mfcc"])
        elif rand > 7:
            sample["mfcc"] = self.time_masking(sample["mfcc"])


        return sample["mfcc"], sample["label"].squeeze()



    def _create_speaker_dataframe(self):
        data_dict = {}
        name_dict = {}
        speaker_labels = os.listdir(self.root_dir)
        for label_id, s in enumerate(speaker_labels):
            audios_filenames = glob.glob(os.path.join(self.root_dir, s, "*.wav"))
            data_dict[label_id] = audios_filenames
            name_dict[label_id] = s

        speaker_df = pd.DataFrame([(key, var) for (key, L) in data_dict.items() for var in L],
                                  columns=['label', 'filename'])

        return speaker_df, name_dict


class SpeakerGammaDataset(Dataset):
    """Speaker gammagram dataset."""

    def __init__(self, root_dir, train=False):
        """
        Args:
            root_dir (string): Directory with all the subdirectory for each speaker and their gammagrams.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if train:
            self.root_dir = os.path.join(root_dir, "train")
        else:
            self.root_dir = os.path.join(root_dir, "test")
        self.speaker_frame, self.name_dict = self._create_speaker_dataframe()


    def __len__(self):
        return len(self.speaker_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feat_name = os.path.join(self.root_dir,
                                  self.speaker_frame.iloc[idx, 1])
        feature = np.load(feat_name)
        feature = torch.from_numpy(feature).float()
        label = self.speaker_frame.iloc[idx, 0]
        label = np.array([label])
        label = label.astype('long')

        return feature, label.squeeze()

    def _create_speaker_dataframe(self):
        data_dict = {}
        name_dict = {}
        speaker_labels = os.listdir(self.root_dir)
        for label_id, s in enumerate(speaker_labels):
            audios_filenames = glob.glob(os.path.join(self.root_dir, s, "*.npy"))
            data_dict[label_id] = audios_filenames
            name_dict[label_id] = s

        speaker_df = pd.DataFrame([(key, var) for (key, L) in data_dict.items() for var in L],
                                  columns=['label', 'filename'])

        return speaker_df, name_dict


class SpeakerDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 128, length_chunk: int = 1000,
                 overlap: int = 500, output_dir: str = "output_dir", melargs=None, feat="raw"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = T.MelSpectrogram(**melargs)

        self.output_dir = output_dir
        self.length_chunk = length_chunk
        self.overlap = overlap
        self.feature = feat

    def prepare_data(self, mean_channels=False) -> None:
        # Create the output dir
        if os.path.exists(self.output_dir):
            return

        os.mkdir(self.output_dir)

        # Process train folder
        tmp_data_dir = os.path.join(self.data_dir, "train")
        tmp_output_dir = os.path.join(self.output_dir, "train")
        process_dataset(tmp_data_dir, tmp_output_dir, self.length_chunk, self.overlap, mean=mean_channels)

        # Process test folder
        tmp_data_dir = os.path.join(self.data_dir, "test")
        tmp_output_dir = os.path.join(self.output_dir, "test")
        process_dataset(tmp_data_dir, tmp_output_dir, self.length_chunk, 1000, mean=mean_channels)

        self.data_dir = self.output_dir

    def setup(self, stage: str = None):
        if self.feature == "raw":
            # Assign Train/val split(s) for use in Dataloaders
            if stage == 'fit' or stage is None:
                ds_full = SpeakerDataset(self.data_dir, train=True, transform=self.transform)
                nb_split_train = int(len(ds_full) * 0.8)
                self.ds_train, self.ds_val = random_split(ds_full,
                                                          [nb_split_train, len(ds_full) - nb_split_train])
            # Assign Test split(s) for use in Dataloaders
            if stage == 'test' or stage is None:
                self.ds_test = SpeakerDataset(self.data_dir, train=False, transform=self.transform)

            self.name_dict = ds_full.name_dict

            self.num_classe = len(self.name_dict.keys())

        elif self.feature == "gammatone":
            # Assign Train/val split(s) for use in Dataloaders
            if stage == 'fit' or stage is None:
                ds_full = SpeakerGammaDataset(self.data_dir, train=True)
                nb_split_train = int(len(ds_full) * 0.8)
                self.ds_train, self.ds_val = random_split(ds_full,
                                                          [nb_split_train, len(ds_full) - nb_split_train])
            # Assign Test split(s) for use in Dataloaders
            if stage == 'test' or stage is None:
                self.ds_test = SpeakerGammaDataset(self.data_dir, train=False)

            self.name_dict = ds_full.name_dict

            self.num_classe = len(self.name_dict.keys())

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=8, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=8, pin_memory=True)

    def get_label_dict(self):
        return self.name_dict
