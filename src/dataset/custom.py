import cv2
import torch
import numpy as np
import albumentations as A
from PIL import Image
from torchvision import transforms
from typing import Optional, Callable
from torchvision.datasets import CIFAR10, CIFAR100, SVHN


class SVHN(SVHN):
    def __init__(
        self,
        root: Optional[str] = "~/data/svhn",
        split: Optional[str] = "train",
        download: Optional[bool] = True,
        transform: Optional[Callable] = None,
    ):
        super(SVHN, self).__init__(root=root, split=split, download=download, transform=transform)

    def __getitem__(self, index: int):
        image = self.data[index].reshape(32, 32, 3)
        image = Image.fromarray(image)
        target = self.labels[index]

        if self.transform:
            if isinstance(self.transform, transforms.Compose):
                image = self.transform(image)
            elif isinstance(self.transform, A.core.composition.Compose):
                image = self.transform(image=image)["image"]

        return {"image": image, "targets": target}


class Cifar10(CIFAR10):
    def __init__(
        self,
        root: Optional[str] = "~/data/cifar10",
        train: Optional[bool] = True,
        download: Optional[bool] = True,
        transform: Optional[Callable] = None,
    ):
        super(Cifar10, self).__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index: int):
        image = self.data[index]
        target = self.targets[index]

        if self.transform:
            if isinstance(self.transform, transforms.Compose):
                image = self.transform(image)
            elif isinstance(self.transform, A.core.composition.Compose):
                image = self.transform(image=image)["image"]

        return {"image": image, "targets": target}


class Cifar100(CIFAR100):
    def __init__(
        self,
        root: Optional[str] = "~/data/cifar100",
        train: Optional[bool] = True,
        download: Optional[bool] = True,
        transform: Optional[Callable] = None,
    ):
        super(Cifar100, self).__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index: int):
        image = self.data[index]
        target = self.targets[index]

        if self.transform:
            if isinstance(self.transform, transforms.Compose):
                image = self.transform(image)
            elif isinstance(self.transform, A.core.composition.Compose):
                image = self.transform(image=image)["image"]

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

        if self.transforms:
            if isinstance(self.transform, transforms.Compose):
                image = self.transform(image)
            elif isinstance(self.transform, A.core.composition.Compose):
                image = self.transform(image=image)["image"]

        if self.targets is None:
            return {"image": image}

        return {"image": image, "targets": torch.tensor(self.targets[index])}
