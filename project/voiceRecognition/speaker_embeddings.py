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
            list_emb = [np.load(emb_f).squeeze() for emb_f in emb_filenames]

            mean = np.array(list_emb).mean(axis=0)
            self.mean_embedding[label_id] = mean
            self.data_dict[label_id] = list_emb
            self.name_dict[label_id] = s

    def get_speaker(self, emb):

        min_score = 0
        final_label = 0
        for speaker_label, mean_emb in self.mean_embedding.items():
            score = self.similarity_func(torch.from_numpy(mean_emb), emb)
            score = score.mean()
            if score > min_score:
                min_score = score
                final_label = speaker_label

        min_score = 0
        for embeddings in self.data_dict[final_label]:
            score = self.similarity_func(torch.from_numpy(embeddings), emb)
            score = score.mean()

            if score > min_score:
                min_score = score

        return min_score, final_label

    def get_name_speaker(self, speaker_id):
        return self.name_dict[speaker_id]


if __name__ == '__main__':
    speaker_emb = SpeakerEmbeddings(OUTPUT_EMB_TRAIN)

    print(speaker_emb.name_dict)


