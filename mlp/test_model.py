import torch
import torchmetrics
import torch.nn.functional as F
from torch import nn
from .scheduler import WarmupCosineLR
from pytorch_lightning import LightningModule, Trainer, LightningDataModule


class CNN(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class LitCIFAR10Model(LightningModule):
    def __init__(self, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.model = CNN()
        self.trainer = trainer

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)
        return loss

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def setup_steps(self, stage=None):
        # After updating to 1.5.2, NotImplementedError: `train_dataloader` · Discussion #10652 · PyTorchLightning/pytorch-lightning https://github.com/PyTorchLightning/pytorch-lightning/discussions/10652
        train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
        return len(train_loader)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
        )
        total_steps = 200 * self.setup_steps(self)
        scheduler = {
            "scheduler": WarmupCosineLR(optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]