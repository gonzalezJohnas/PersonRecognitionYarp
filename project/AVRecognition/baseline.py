from torch.utils.data import Dataset, DataLoader, random_split

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Softmax, AvgPool2d
from project.AVRecognition.PersonDataModule import PersonDataModule, PersonDataset
import torchvision.models as models
from collections import OrderedDict
from project.voiceRecognition.speaker_embeddings import SpeakerEmbeddings
from project.AVRecognition.lit_AVperson_classifier import LitSpeakerClassifier, Backbone
from tqdm import tqdm
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

threshold = 0.42


def max_Score(score_1, score_2, label_1, label_2):
    if score_1 > score_2:
        return label_1
    else:
        return label_2

def cli_main():
    pl.seed_everything(1234)

    # ------------
    # data
    # ------------
    parameters = {
        "data_dir": "/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/audio_visual/",
        "batch_size": 1,
    }


    model = torch.load("/home/icub/PycharmProjects/SpeakerRecognitionYarp/project/AVRecognition/model_audiovisual.pt")
    sm = torch.nn.Softmax(dim=1)

    db_embeddings_voices = SpeakerEmbeddings("/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/voice/dataset_emb_vad_2s/train")
    db_embeddings_faces = SpeakerEmbeddings("/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/face/train")

    ds_test = PersonDataset(parameters["data_dir"], train=False, concat=False)
    test_loader = DataLoader(ds_test, batch_size=1, num_workers=8, pin_memory=True)
    num_class = len(ds_test.name_dict.keys())
    confusion_matrix_baseline = np.zeros((num_class, num_class))
    confusion_matrix_model = np.zeros((num_class, num_class))

    print(f"Number of class {num_class} total samples {len(test_loader)}")
    with tqdm(total=len(test_loader)) as pbar:
        for emb_test, label_test in test_loader:
            score_voice, predicted_label_voice = db_embeddings_voices.get_speaker(emb_test[0])
            score_face, predicted_label_face = db_embeddings_faces.get_speaker(emb_test[1])

            full_emb = np.hstack((emb_test[0], emb_test[1]))
            full_emb = torch.from_numpy(full_emb).cuda()
            outputs = sm(model(full_emb))
            prediction_model = torch.argmax(outputs.cpu())

            if score_voice > threshold or score_face > threshold:
                predicted_label = max_Score(score_voice, score_face, predicted_label_voice, predicted_label_face)
                confusion_matrix_baseline[label_test, predicted_label] += 1
                confusion_matrix_model[label_test, prediction_model] += 1
            pbar.update(1)

        class_names = [ds_test.name_dict[i] for i in range(0, num_class)]

        total_accuracy_baseline = confusion_matrix_baseline.diagonal().sum() / len(test_loader)
        confusion_matrix_baseline_n = confusion_matrix_baseline.astype('float') / confusion_matrix_baseline.sum(axis=1)[:, np.newaxis]

        df_cm_baseline = pd.DataFrame(confusion_matrix_baseline_n, index=class_names, columns=class_names)
        plt.figure(figsize=(18, 15))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm_baseline, annot=True, annot_kws={"size": 16})  # font size

        plt.savefig("confusion_matrix_baseline.png")
        print(f"Baseline Total accuracy {total_accuracy_baseline}%")
        plt.figure(figsize=(18, 15))

        total_accuracy_model = confusion_matrix_model.diagonal().sum() / len(test_loader)
        confusion_matrix_model_n = confusion_matrix_model.astype('float') / confusion_matrix_model.sum(axis=1)[:, np.newaxis]

        df_cm_model = pd.DataFrame(confusion_matrix_model_n, index=class_names, columns=class_names)
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm_model, annot=True, annot_kws={"size": 16})  # font size
        plt.savefig("confusion_matrix_model.png")

        print(f"Model Total accuracy {total_accuracy_model}%")


if __name__ == '__main__':
    cli_main()
