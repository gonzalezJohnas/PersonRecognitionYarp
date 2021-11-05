import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from utils import face_alignement
import os, glob
import pandas as pd
import torchaudio
import torch
import seaborn as sns
import tqdm
import time
import cv2
from torchvision import datasets, transforms
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from PIL import Image

PATH_TRAIN = "/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/face/train"
PATH_TEST = "/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/face/test"
PATH_TEST_UNKNOWN = "/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/face/unknown_person"
PATH_MODEL = "/home/icub/PycharmProjects/SpeakerRecognitionYarp/project/faceRecognition/saved_model/model_triple_facerecogntion.pt"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OUTPUT_EMB_TRAIN = "/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/voice/dataset_emb/"
encoder = InceptionResnetV1(
        classify=False,
        pretrained="vggface2",
        num_classes=12,
        dropout_prob=0.5
    ).to(device)
encoder.eval()


threshold = 0.7


class FaceDataset(Dataset):
    """Speaker raw audio dataset."""

    def __init__(self, root_dir, transform=None, extension="*.npy"):
        """
        Args:
            root_dir (string): Directory with all the subdirectory for each face and their image.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.file_extension = extension
        self.face_frame, self.name_dict = self._create_speaker_dataframe()
        self.transform = transform

    def __len__(self):
        return len(self.face_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.file_extension == "*.npy":
            file_name = os.path.join(self.root_dir,
                                     self.face_frame.iloc[idx, 1])
            sample = np.load(file_name)
            sample = torch.from_numpy(sample)


        else:
            return None, None

        label = self.face_frame.iloc[idx, 0]
        label = np.array([label])
        label = label.astype('long')

        return sample, label[0]

    def _create_speaker_dataframe(self):
        data_dict = {}
        name_dict = {}
        face_labels = os.listdir(self.root_dir)
        for label_id, s in enumerate(face_labels):
            faces_filenames = glob.glob(os.path.join(self.root_dir, s, self.file_extension))
            data_dict[label_id] = faces_filenames
            name_dict[label_id] = s

        face_df = pd.DataFrame([(key, var) for (key, L) in data_dict.items() for var in L],
                                  columns=['label', 'filename'])

        return face_df, name_dict


class FacesEmbeddings:

    def __init__(self, dataset_dir):
        self.root_dir = dataset_dir
        self.mean_embedding = {}
        self.data_dict = {}
        self.name_dict = {}

        self._load_dataset()

        self.similarity_func = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def _load_dataset(self):

        speaker_labels = os.listdir(self.root_dir)
        for label_id, s in enumerate(speaker_labels):
            emb_filenames = glob.glob(os.path.join(self.root_dir, s, "*.npy"))
            list_emb = [np.load(emb_f).squeeze() for emb_f in emb_filenames]

            mean = np.array(list_emb).mean(axis=0)
            self.mean_embedding[label_id] = mean
            self.data_dict[label_id] = list_emb
            self.name_dict[label_id] = s

    def get_speaker(self, emb):

        min_score = 0
        final_label = 0
        for speaker_label, list_emb in self.data_dict.items():
            for embt in list_emb:
                score = self.similarity_func(torch.from_numpy(embt), emb).numpy()
                if score > min_score:
                    min_score = score
                    final_label = speaker_label

        # min_score = 0
        # for embeddings in self.data_dict[final_label]:
        #     score = self.similarity_func(torch.from_numpy(embeddings), emb).numpy()
        #     if score > min_score:
        #         min_score = score

        return min_score, final_label

    def get_name_speaker(self, speaker_id):
        return self.name_dict[speaker_id]



def get_embeddings(dataset, encoder, gen=False):

    X = []
    y = []

    for x, label in dataset:
        if gen:
            emb = encoder.encode_batch(x)
        else:
            emb = x
        X.append(emb)
        y.append(label)

    return X, y


def test_negative_accuracy(train_loader):

    test_loader = FaceDataset(PATH_TEST_UNKNOWN)

    X_test, y_test = get_embeddings(test_loader, encoder)
    nb_matched = 0

    for emb_test, label_test in zip(X_test, y_test):
        score, predicted_label = train_loader.get_speaker(emb_test)
        print(score)
        if score < threshold:
            nb_matched += 1

    print(f"Negative accuracy {nb_matched/len(X_test)}")




def get_tensor_from_image(img_path, trans):
    frame = Image.open(img_path)
    tensor = trans(frame).unsqueeze(0).cuda(0)

    return tensor


def create_embeddings(data_dir, output_dir):
    """
    :param data_dir: Root dir of the dataset of speakers
    :param output_dir: Path to the output_dir
    :return:
    """

    # Process train folder
    dirs = os.listdir(data_dir)

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
        transforms.Resize((180, 180))
    ])

    for people_dir in dirs:
        list_img_filenames = []
        for ext in ('*.png', '*.jpg'):
            list_img_filenames.extend( glob.glob(os.path.join(data_dir, people_dir, ext)))

        for i, img_path in enumerate(list_img_filenames):
            input_tensor = get_tensor_from_image(img_path, trans)
            embeddings = encoder(input_tensor).data.cpu().numpy()
            enbeddings_path = os.path.join(data_dir, people_dir) + f"/face_{i}emb.npy"
            np.save(enbeddings_path, embeddings.ravel())



def main(train_loader):

    test_loader = FaceDataset(PATH_TEST)

    X_test, y_test = get_embeddings(test_loader, encoder)

    name_dict = test_loader.name_dict

    confusion_matrix = np.zeros((len(name_dict), len(name_dict)))

    for emb_test, label_test in zip(X_test, y_test):
        score, predicted_label = train_loader.get_speaker(emb_test)
        print(score)
        if score > threshold:
            confusion_matrix[label_test, predicted_label] += 1


    plt.figure(figsize=(15,10))

    # Per-class accuracy
    # class_accuracy = 100 * confusion_matrix.diagonal() /
    total_accuracy = confusion_matrix.diagonal().sum() / len(test_loader)
    print(f"Total accuracy {total_accuracy}%")
    confusion_matrix_n = confusion_matrix.astype('float')  / confusion_matrix.sum(axis=1)[:,
                                                                              np.newaxis]

    class_names = [name_dict[i] for i in range(0, len(name_dict))]
    df_cm = pd.DataFrame(confusion_matrix_n, index=class_names, columns=class_names)
    heatmap = sns.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='.2f',  cbar_kws={'format': '%.0f%%', 'ticks': [0, 100]})

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.title("Confusion matrix  Triplet-Loss, Positive accuracy {}%".format(round(total_accuracy, 2)), fontsize=20)
    plt.savefig("confusion_matrix_embeddings.png")


if __name__ == "__main__":
    create_embeddings(PATH_TEST, PATH_TEST)
    create_embeddings(PATH_TRAIN, PATH_TRAIN)

    t = time.process_time()

    train_loader = FacesEmbeddings(PATH_TRAIN)
    end = time.time()
    print(f"Elapsed time {time.process_time() - t}")

    main(train_loader)
    print("Negative accuracy processing")
    test_negative_accuracy(train_loader)