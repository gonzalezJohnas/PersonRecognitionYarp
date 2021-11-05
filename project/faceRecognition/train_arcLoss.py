from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import pytorch_metric_learning

import numpy as np
import os
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import umap
from cycler import cycler


logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s"%pytorch_metric_learning.__version__)
torch.manual_seed(1234)

# TRAIN PARAMETERS
data_dir = '/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/face'
model_path = '/home/icub/PycharmProjects/SpeakerRecognitionYarp/project/faceRecognition/saved_model' \
             '/model_triple_facerecogntion_144v2.pt'
LR = 1e-3
workers = 0 if os.name == 'nt' else 8
margin_p = 0.3
# Set other training parameters
batch_size = 80
num_epochs = 150



def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))



    # Data transformations
    trans_train = transforms.Compose([
        transforms.RandomApply(transforms=[
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomHorizontalFlip(),
        ]),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
    ])

    trans_val = transforms.Compose([
        # transforms.CenterCrop(120),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train_aligned"), transform=trans_train)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val_aligned"), transform=trans_val)

    # Prepare the model
    model = InceptionResnetV1(
        classify=False,
        pretrained="vggface2",
        dropout_prob=0.5
    ).to(device)

    # for param in list(model.parameters())[:-8]:
    #     param.requires_grad = False



    trunk_optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # Set the loss function
    loss = losses.ArcFaceLoss(len(train_dataset.classes), 512)



    # Package the above stuff into dictionaries.
    models = {"trunk": model}
    optimizers = {"trunk_optimizer": trunk_optimizer}
    loss_funcs = {"metric_loss": loss}
    mining_funcs = {}
    lr_scheduler = {"trunk_scheduler_by_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau(trunk_optimizer)}


    # Create the tester
    record_keeper, _, _ = logging_presets.get_record_keeper("logs", "tensorboard")
    hooks = logging_presets.get_hook_container(record_keeper)

    dataset_dict = {"val": val_dataset, "train": train_dataset}
    model_folder = "training_saved_models"

    def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
        logging.info("UMAP plot for the {} split and label set {}".format(split_name, keyname))
        label_set = np.unique(labels)
        num_classes = len(label_set)
        fig = plt.figure(figsize=(8, 7))
        plt.gca().set_prop_cycle(cycler("color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]))
        for i in range(num_classes):
            idx = labels == label_set[i]
            plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
        plt.show()

    tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook=hooks.end_of_testing_hook,
                                                dataloader_num_workers=4,
                                                accuracy_calculator=AccuracyCalculator(include=['mean_average_precision_at_r'], k="max_bin_count"))


    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder, splits_to_eval=[('val',['train'])])

    # Create the trainer
    trainer = trainers.MetricLossOnly(models,
                                      optimizers,
                                      batch_size,
                                      loss_funcs,
                                      mining_funcs,
                                      train_dataset,
                                      lr_schedulers=lr_scheduler,
                                      dataloader_num_workers=8,
                                      end_of_iteration_hook=hooks.end_of_iteration_hook,
                                      end_of_epoch_hook=end_of_epoch_hook)

    trainer.train(num_epochs=num_epochs)


if __name__ == '__main__':
    main()

