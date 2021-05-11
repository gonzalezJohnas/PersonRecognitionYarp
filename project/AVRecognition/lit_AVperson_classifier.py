from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Softmax, AvgPool2d
from project.AVRecognition.PersonDataModule import PersonDataModule
import torchvision.models as models
from collections import OrderedDict


class Backbone(torch.nn.Module):
    def __init__(self, nb_class=13):
        super().__init__()
        self.relu = ReLU()
        self.dropout = torch.nn.Dropout()
        self.l1 = torch.nn.Linear(704, 256)
        self.l2 = torch.nn.Linear(256, 64)
        self.l3 = torch.nn.Linear(64, 32)

        self.output = torch.nn.Linear(32, nb_class)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class LitSpeakerClassifier(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)

        preds = self.sm(y_hat)
        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

        preds = self.sm(y_hat)
        self.valid_acc(preds, y)

        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        preds = self.sm(y_hat)
        self.test_acc(preds, y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-2)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.train_acc.compute())
        print(f"\nTrain Accuracy {self.train_acc.compute()}")
        print(f"\nVal Accuracy {self.valid_acc.compute()}")


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitSpeakerClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    parameters = {
        "data_dir": "/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/audio_visual/",
        "batch_size": args.batch_size,
    }

    dm = PersonDataModule(**parameters)
    dm.setup()

    print(f"Number of class {dm.num_classe}")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # ------------
    # model
    # ------------

    model = LitSpeakerClassifier(Backbone(nb_class=dm.num_classe), args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)

    torch.save(model, "model_audiovisual.pt")


if __name__ == '__main__':
    cli_main()
