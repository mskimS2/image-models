import torch
import numpy as np
from torch import nn
from sklearn import metrics
from torchvision import transforms

from dataset.custom import Cifar10, Cifar100, SVHN
from trainer.custom.config import Config
from trainer.custom.trainer import Trainer
from transform.auto_aug import AutoAugment
from models import get_model


class ImageClassifier(nn.Module):
    def __init__(self, model):
        super(ImageClassifier, self).__init__()
        self.model = model
        self._init_weights()

    def monitor_metrics(self, outputs, targets):
        device = targets.device.type
        outputs = torch.argmax(outputs, dim=-1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        f1 = metrics.f1_score(targets, outputs, average="macro")
        acc = metrics.accuracy_score(targets, outputs)
        recall = metrics.recall_score(targets, outputs, average="macro")
        precision = metrics.precision_score(targets, outputs, average="macro")

        return {
            "f1": torch.tensor(f1, device=device),
            "acc": torch.tensor(acc, device=device),
            "recall": torch.tensor(recall, device=device),
            "precision": torch.tensor(precision, device=device),
        }

    def fetch_optimizer_and_scheduler(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=-0.5, patience=2, verbose=True, mode="max", threshold=1e-4
        )

        return optimizer, scheduler

    def loss_function(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)

    def forward(self, image, targets=None):
        outputs = self.model(image)
        if targets is not None:
            loss = self.loss_function(outputs, targets)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics

        return outputs, 0, {}

    def _init_weights(self, pretrained: str = None) -> None:
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if n.startswith("head"):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def get_dataset_and_dataloader(config: Config):
    train_aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]
    )

    test_aug = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]
    )

    match config.dataset_name:
        case "cifar10":
            train_dataset = Cifar10(root="./data/", split="train", download=True, transform=train_aug)
            test_dataset = Cifar10(root="./data/", split="test", download=True, transform=test_aug)

        case "cifar100":
            train_dataset = Cifar100(root="./data/", train=True, download=True, transform=train_aug)
            test_dataset = Cifar100(root="./data/", train=False, download=True, transform=test_aug)

        case "svhn":
            train_dataset = SVHN(root="./data/", split="train", download=True, transform=train_aug)
            test_dataset = SVHN(root="./data/", split="test", download=True, transform=test_aug)

        case _:
            raise ValueError("dataset_name must be `cifar10`, `cifar100`, `svhn`")

    return train_dataset, test_dataset


if __name__ == "__main__":

    config = Config(
        training_batch_size=128,
        validation_batch_size=128,
        epochs=300,
        step_scheduler_after="epoch",
        step_scheduler_metric="valid_f1",
        model_name="mlpmixer",
        num_workers=4,
        dataset_name="cifar10",
    )

    train_dataset, test_dataset = get_dataset_and_dataloader(config)

    model = get_model(config)
    model.to(config.device)

    classfier = ImageClassifier(model)
    classfier = Trainer(classfier)

    classfier.fit(train_dataset, valid_dataset=test_dataset, config=config)
