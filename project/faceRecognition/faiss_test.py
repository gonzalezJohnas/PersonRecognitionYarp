from torchvision import transforms, datasets
import torch
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder, CosineSimilarity
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
import numpy as np
from project.faceRecognition.utils import fixed_image_standardization

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Parameters
model_path = '/home/icub/PycharmProjects/SpeakerRecognitionYarp/project/faceRecognition/saved_model' \
             '/model_triple_facerecogntion_115.pt'
database_path = '/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/face/train_aligned'
test_set_path = '/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/face/test_aligned'

transform_pipeline = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization,
])

nb_neighbors = 50
distance_threshold = 1.0


def indices_to_class(predicted_indices, indices_class):
    ans = []
    for ind in predicted_indices:
        for k in indices_class:
            if ind in indices_class[k]:
                ans.append(k)
                break
    return ans


def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path).to(device)
    match_finder = MatchFinder(CosineSimilarity(), threshold=0.4)
    inference_model = InferenceModel(model, match_finder=match_finder)

    faces_db = datasets.ImageFolder(database_path, transform=transform_pipeline)
    faces_test = datasets.ImageFolder(test_set_path, transform=transform_pipeline)
    name_dict = faces_db.classes

    inference_model.train_indexer(faces_db)
    labels_to_indices = c_f.get_labels_to_indices(faces_db.targets)

    confusion_matrix = np.zeros((len(name_dict)+1, len(name_dict)+1))

    for img, label in faces_test:
        img = img.unsqueeze_(0).to(device)

        indices, distances = inference_model.get_nearest_neighbors(img, k=nb_neighbors)
        distances = distances[0]
        indices = indices[0]
        filter_indices = indices[distances < distance_threshold]

        class_name = faces_test.classes[label]
        if class_name == 'unknown':
            true_label = len(name_dict)
        else:
            true_label = faces_db.class_to_idx[class_name]
        if len(filter_indices):
            class_neighbors = indices_to_class(filter_indices, labels_to_indices)
            prediction_index = max(class_neighbors, key=class_neighbors.count)
            confusion_matrix[true_label, prediction_index] += 1
        else:
            confusion_matrix[true_label,  len(name_dict)] += 1


    plt.figure(figsize=(15, 10))

    # Total accuracy
    confusion_matrix_n = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:,
                                                            np.newaxis]
    confusion_matrix_n[np.isnan(confusion_matrix_n)] = 0

    positive_accuracy = confusion_matrix_n.diagonal()[:-1].sum() / (len(faces_test.class_to_idx)-1)
    print(f"Positive accuracy {positive_accuracy}%")

    negative_accuracy = confusion_matrix_n.diagonal()[-1]
    print(f"Negative accuracy {negative_accuracy}%")

    class_names = [name_dict[i] for i in range(0, len(name_dict))] + ['unknown']
    df_cm = pd.DataFrame(confusion_matrix_n, index=class_names, columns=class_names)
    heatmap = sns.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='.2f',
                          cbar_kws={'format': '%.0f%%', 'ticks': [0, 100]})

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.title(
        f"Confusion matrix  Triplet-Loss, Positive - Negative accuracy \
        ({round(positive_accuracy, 2)}%, {round(negative_accuracy, 2)}%) ", fontsize=20)
    plt.savefig("confusion_matrix.png")

if __name__ == "__main__":
    main()