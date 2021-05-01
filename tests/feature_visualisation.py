import matplotlib
import matplotlib.pyplot as plt
from project.SpeakerDataModule import SpeakerDataset, SpeakerDataModule
import librosa

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)

if __name__ == "__main__":

    parameters={
        "data_dir" : "/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/datasetv2",
        "length_chunk" : 1000,
        "overlap" : 500,
        "output_dir" : "/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/datasetv2",
        "melargs" : {'n_fft': 2048, 'n_mels': 256, 'hop_length': 512}
    }

    dm = SpeakerDataModule(**parameters)
    dm.setup('fit')
    a = dm.train_dataloader()

    for sample in a:
        mfcc = sample["mfcc"]
        label = sample["label"]
        plot_spectrogram(mfcc[0].numpy()[0])