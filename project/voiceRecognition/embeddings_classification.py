from speechbrain.pretrained import EncoderClassifier
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import os, glob
import pandas as pd
import torchaudio
import torch
import seaborn as sns
import tqdm
from speaker_embeddings import SpeakerEmbeddings
import time

PATH_TRAIN = "/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/voice/dataset_emb_vad_2/train"
PATH_TEST = "/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/voice/dataset_vad/test"
PATH_TEST_UNKNOWN = "/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/voice/raw/test_unknown"

OUTPUT_EMB_TRAIN = "/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/voice/dataset_emb/"
encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

threshold = 0.42


class SpeakerDataset(Dataset):
    """Speaker raw audio dataset."""

    def __init__(self, root_dir, transform=None, extension="*.wav"):
        """
        Args:
            root_dir (string): Directory with all the subdirectory for each speaker and their audio.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.file_extension = extension
        self.speaker_frame, self.name_dict = self._create_speaker_dataframe()
        self.transform = transform
        self.sample_rate = 16000
        self.resample_trans = torchaudio.transforms.Resample(48000, self.sample_rate)

    def __len__(self):
        return len(self.speaker_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.file_extension == "*.wav":
            audio_name = os.path.join(self.root_dir,
                                      self.speaker_frame.iloc[idx, 1])
            sample, sample_rate = torchaudio.load(audio_name)
            sample = self.resample_trans(sample)

        elif self.file_extension == "*.npy":
            file_name = os.path.join(self.root_dir,
                                     self.speaker_frame.iloc[idx, 1])
            sample = np.load(file_name)
            sample = torch.from_numpy(sample)

        else:
            return None, None

        label = self.speaker_frame.iloc[idx, 0]
        label = np.array([label])
        label = label.astype('long')

        return sample, label[0]

    def _create_speaker_dataframe(self):
        data_dict = {}
        name_dict = {}
        speaker_labels = os.listdir(self.root_dir)
        for label_id, s in enumerate(speaker_labels):

            audios_filenames = glob.glob(os.path.join(self.root_dir, s, self.file_extension))
            data_dict[label_id] = audios_filenames
            name_dict[label_id] = s

        speaker_df = pd.DataFrame([(key, var) for (key, L) in data_dict.items() for var in L],
                                  columns=['label', 'filename'])

        return speaker_df, name_dict


def create_embeddings(data_dir, output_dir):
    """
    :param data_dir: Root dir of the dataset of speakers
    :param output_dir: Path to the output_dir
    :return:
    """

    # Process train folder
    dirs = os.listdir(data_dir)
    os.mkdir(output_dir)
    resample_trans = torchaudio.transforms.Resample(48000, 16000)

    for d in dirs:
        print(f"Processing directory {d}")
        audio_files = glob.glob(os.path.join(data_dir, d, "*.wav"))
        os.mkdir(os.path.join(output_dir, d))
        for i, f in tqdm.tqdm(enumerate(audio_files)):
            waveform, sample_rate = torchaudio.load(f)
            waveform = resample_trans(waveform)
            emb = encoder.encode_batch(waveform)
            output_filename = os.path.join(output_dir, d, f"emb_{i}.npy")
            np.save(output_filename, emb)


def get_embeddings(dataset, encoder):

    X = []
    y = []

    for x, label in dataset:
        emb = encoder.encode_batch(x)
        X.append(emb)
        y.append(label)

        if len(X)==2:
            break
    return X, y


def test_negative_accuracy(train_loader):

    test_loader = SpeakerDataset(PATH_TEST_UNKNOWN)

    X_test, y_test = get_embeddings(test_loader, encoder)
    nb_matched = 0

    for emb_test, label_test in zip(X_test, y_test):
        score, predicted_label = train_loader.get_speaker(emb_test)
        print(score)
        if score < threshold:
            nb_matched += 1

    print(f"Negative accuracy {nb_matched/len(X_test)}")





def main(train_loader):

    test_loader = SpeakerDataset(PATH_TEST)

    X_test, y_test = get_embeddings(test_loader, encoder)


    name_dict = test_loader.name_dict

    confusion_matrix = np.zeros((len(name_dict), len(name_dict)))
    for emb_test, label_test in zip(X_test, y_test):
        score, predicted_label = train_loader.get_speaker(emb_test)
        score = score.mean()

        if score > threshold:
            confusion_matrix[label_test, predicted_label] += 1


    plt.figure(figsize=(15,10))

    # Per-class accuracy
    # class_accuracy = 100 * confusion_matrix.diagonal() /
    total_accuracy = confusion_matrix.diagonal().sum() / len(test_loader)
    print(f"Total accuracy {total_accuracy}%")
    confusion_matrix_n = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:,
                                                                              np.newaxis]
    class_names = [name_dict[i] for i in range(0, len(name_dict))]
    df_cm = pd.DataFrame(confusion_matrix_n, index=class_names, columns=class_names)
    heatmap = sns.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='.2f',  cbar_kws={'format': '%.0f%%', 'ticks': [0, 100]})

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.title("Confusion matrix  Triplet-Loss, Positive accuracy {}%".format(round(total_accuracy, 2)), fontsize=20 )
    plt.savefig("confusion_matrix_embeddings.png")


if __name__ == "__main__":
    # create_embeddings(PATH_TEST, "/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/audio_visual/test")

    test_loader = SpeakerDataset(PATH_TEST)

    x, label = test_loader[0]

    t = time.process_time()

    train_loader = SpeakerEmbeddings(OUTPUT_EMB_TRAIN)
    end = time.time()
    print(f"Elapsed time {time.process_time() - t}")

    main(train_loader)
    print("Negative accuracy processing")
    test_negative_accuracy(train_loader)