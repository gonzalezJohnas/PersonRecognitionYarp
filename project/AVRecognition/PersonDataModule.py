import itertools

from torch.utils.data import Dataset, DataLoader, random_split
import os, glob
import pandas as pd
import pytorch_lightning as pl
import torchaudio, torch
import numpy as np
import torchaudio.transforms as T


class PersonDataset(Dataset):
    """Speaker raw audio dataset."""

    def __init__(self, root_dir, transform=None, train=False, concat=True):
        """
        Args:
            root_dir (string): Directory with all the subdirectory for each speaker and their embeddings.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if train:
            self.root_dir = os.path.join(root_dir, "train")
        else:
            self.root_dir = os.path.join(root_dir, "test")
        self.people_frame, self.name_dict = self._create_speaker_dataframe()
        self.transform = transform
        self.concat = concat

    def __len__(self):
        return len(self.people_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        voice_emb_filename = os.path.join(self.root_dir,
                                          self.people_frame.iloc[idx, 0])
        face_emb_filename = os.path.join(self.root_dir,
                                         self.people_frame.iloc[idx, 1])

        voice_emb = np.load(voice_emb_filename)
        face_emb = np.load(face_emb_filename)

        if self.concat:
            sample = np.hstack((voice_emb.squeeze(), face_emb))
        else:
            sample = [voice_emb.squeeze(), face_emb]
        label = self.people_frame.iloc[idx, 2]
        label = np.array([label])
        label = label.astype('long')

        return sample, label.squeeze()

    def _create_speaker_dataframe(self):

        person_df = pd.DataFrame([],
                                 columns=['emv_voice_filename', 'emv_face_filename', 'label'])

        name_dict = {}
        speaker_labels = os.listdir(self.root_dir)
        for label_id, s in enumerate(speaker_labels):
            list_embeddings = glob.glob(os.path.join(self.root_dir, s, "*.npy"))
            face_emb = []
            voice_emb = []
            for emb_name in list_embeddings:
                if "face" in emb_name:
                    face_emb.append(emb_name)
                else:
                    voice_emb.append(emb_name)

            list_combination_emb = list(itertools.product(voice_emb, face_emb))
            tmp_df = pd.DataFrame(list_combination_emb, columns=['emv_voice_filename', 'emv_face_filename'])
            tmp_df["label"] = label_id
            name_dict[label_id] = s


            person_df = person_df.append(tmp_df, ignore_index=True)

        return person_df, name_dict


class PersonDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = None

    def setup(self, stage: str = None):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit' or stage is None:
            ds_full = PersonDataset(self.data_dir, train=True, transform=self.transform)
            nb_split_train = int(len(ds_full) * 0.8)
            self.ds_train, self.ds_val = random_split(ds_full,
                                                      [nb_split_train, len(ds_full) - nb_split_train])
        # Assign Test split(s) for use in Dataloaders
        if stage == 'test' or stage is None:
            self.ds_test = PersonDataset(self.data_dir, train=False, transform=self.transform)

        self.name_dict = ds_full.name_dict

        self.num_classe = len(self.name_dict.keys())


    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=8, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=8, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=8, pin_memory=True)

    def get_label_dict(self):
        return self.name_dict
