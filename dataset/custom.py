import cv2
import torch
import numpy as np
from PIL import Image
from typing import Optional
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from pytorch_lightning import LightningDataModule


class SVHN(SVHN):
    def __init__(
        self,
        root: Optional[str] = "~/data/svhn",
        split: Optional[str] = "train",
        download: Optional[bool] = True,
        transform: Optional[bool] = None
    ):
        super(SVHN, self).__init__(
            root=root,
            split=split,
            download=download,
            transform=transform
        )

    def __getitem__(self, index: int):
        image = self.data[index].reshape(32, 32, 3)
        image = Image.fromarray(image)
        target = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "targets": target}


class Cifar10(CIFAR10):
    def __init__(
        self,
        root: Optional[str] = "~/data/cifar10",
        train: Optional[bool] = True,
        download: Optional[bool] = True,
        transform: Optional[bool] = None
    ):
        super(Cifar10, self).__init__(
            root=root,
            train=train,
            download=download,
            transform=transform
        )

    def __getitem__(self, index: int):
        image = self.data[index]
        target = self.targets[index]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "targets": target}


class Cifar100(CIFAR100):
    def __init__(
        self,
        root: Optional[str] = "~/data/cifar100",
        train: Optional[bool] = True,
        download: Optional[bool] = True,
        transform: Optional[bool] = None
    ):
        super(Cifar100, self).__init__(
            root=root,
            train=train,
            download=download,
            transform=transform
        )

    def __getitem__(self, index: int):
        image = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        return {"image": image, "targets": target}


class ImageDataset:
    def __init__(
        self,
        image_paths: Optional[list],
        targets: Optional[np.array] = None,
        transforms=None,
    ):
        self.image_paths = image_paths
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image_tensor = self.transforms(image=image)["image"]

        # test mode
        if targets is None:
            return {"image": image_tensor}

        # training mode
        targets = torch.tensor(self.targets[index])
        return {"image": image_tensor, "targets": targets}