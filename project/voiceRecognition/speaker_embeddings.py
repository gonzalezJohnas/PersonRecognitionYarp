import numpy as np
import os, glob
import torch


OUTPUT_EMB_TRAIN = "/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/dataset_emb/train"



class SpeakerEmbeddings:

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
            list_emb = [np.load(emb_f) for emb_f in emb_filenames]

            mean = self._get_mean_embedding(list_emb)
            self.mean_embedding[label_id] = mean
            self.data_dict[label_id] = list_emb
            self.name_dict[label_id] = s

    def _get_mean_embedding(self, embeddings):
        mean_emb = []
        for emb in embeddings:
            mean_emb.append(emb.mean(axis=0))

        return np.array(mean_emb).mean(axis=0)

    def get_speaker(self, emb):
        min_score = 0
        final_label = 0
        for speaker_label, mean_emb in self.mean_embedding.items():
            score = self.similarity_func(torch.from_numpy(mean_emb), emb)
            if score[0] > min_score:
                min_score = score[0]
                final_label = speaker_label

        min_score = 0
        for embeddings in self.data_dict[final_label]:
            score = self.similarity_func(torch.from_numpy(embeddings), emb)
            if score[0] > min_score:
                min_score = score[0]

        return min_score, final_label


if __name__ == '__main__':
    speaker_emb = SpeakerEmbeddings(OUTPUT_EMB_TRAIN)

    print(speaker_emb.name_dict)


