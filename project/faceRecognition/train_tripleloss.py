from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


from triplettorch import TripletDataset, HardNegativeTripletMiner, AllTripletMiner

import numpy as np
import os
import pickle
from tqdm import tqdm

torch.manual_seed(17)

# TRAIN PARAMETERS
data_dir = './Database_v2'
batch_size = 64
epochs = 20
LR = 1e-4
workers = 0 if os.name == 'nt' else 8


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))


    # Data transformations
    trans_train = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.RandomApply([
            transforms.RandomCrop(180),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.5, 0.2, 0.1),
        ]),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
    ])

    train_set = datasets.ImageFolder(data_dir, transform=trans_train)
    # Create Lambda function to access data in correct format
    get_data_fn = lambda index: train_set[index][0].float().numpy()

    # Triplet Dataset Definition
    tri_train_set = TripletDataset(train_set.targets, get_data_fn, len(train_set.targets), 6)

    # Data Loader
    tri_train_load = DataLoader(tri_train_set,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2,
                                pin_memory=True
                                )

    miner = HardNegativeTripletMiner(.6).cuda()  # AllTripletMiner( .5 ).cuda( )



    # Prepare the model
    model = InceptionResnetV1(
        classify=False,
        pretrained="vggface2",
        num_classes=len(train_set.class_to_idx),
        dropout_prob=0.5
    ).to(device)

    # Freeze al layers except the last linear layer for re-training
    for param in model.parameters():
        param.requires_grad = False
    model.last_linear.requires_grad_(True)


    optim = torch.optim.Adam(model.parameters(), lr=LR)


    # Train model - Main loop
    for e in tqdm(range(epochs), desc='Epoch'):
        # ================== TRAIN ========================
        train_n = len(tri_train_load)
        train_loss = 0.
        train_frac_pos = 0.

        with tqdm(tri_train_load, desc='Batch') as b_pbar:
            for b, batch in enumerate(b_pbar):
                optim.zero_grad()

                labels, data = batch
                labels = torch.cat([label for label in labels], axis=0)
                data = torch.cat([datum for datum in data], axis=0)
                labels = labels.cuda()
                data = data.cuda()

                embeddings = model(data)
                loss, frac_pos = miner(labels, embeddings)

                loss.backward()
                optim.step()

                train_loss += loss.detach().item()
                train_frac_pos += frac_pos.detach().item() if frac_pos is not None else \
                    0.

                b_pbar.set_postfix(
                    train_loss=train_loss / train_n,
                    train_frac_pos=f'{(train_frac_pos / train_n):.2%}'
                )

    torch.save(model, "saved_model/model_triple_facerecogntion.pt")


if __name__ == '__main__':
    main()

