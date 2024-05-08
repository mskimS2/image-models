from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torch.utils.data import random_split, DataLoader
from typing import Any, Dict
from torchvision import transforms as T


class ImageDataModule(LightningDataModule):
    trainer: Trainer
    
    def __init__(self, config, dataset) -> None:
        super(ImageDataModule, self).__init__()
        
        self.save_hyperparameters(config)
        self.dataset = dataset
        
    def setup(self, _stage: str = None):
        self.dataset(split="train")
        self.dataset(split="validation")
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset(transforms, split="train"),
            batch_size=self.hparams.batch_size, 
            drop_last=True, 
            shuffle=True,
            num_workers=0 if self.on_device else self.trainer.num_devices*4,
            prefetch_factor=2,
            persistant_workers=True,
            pin_memory=not self.on_device,
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset(transforms, split="validation"), 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            drop_last=True, 
            num_workers=0 if self.on_device else self.trainer.num_devices*4,
            prefetch_factor=2,
            persistant_workers=True,
            pin_memory=not self.on_device,
        )
    
    @property
    def on_device(self) -> bool:
        return isinstance(self.trainer.accelerator, CPUAccelerator)
        

class CIFAR10DataModule(LightningDataModule):
    name: str = "cifar10"
    
    def __init__(self, batch_size: int, data_dir: str = "~/data/cifar10", transform = None, ratio: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_classes = 10
        self.ratio = ratio
    
    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            train_ratio = int(len(cifar_full) * (1 - self.ratio))
            val_ratio  = len(cifar_full) - train_ratio
            self.cifar_train, self.cifar_val = random_split(cifar_full, [train_ratio, val_ratio])

        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)
    
    
class CIFAR100DataModule(LightningDataModule):
    name: str = "cifar100"
    
    def __init__(self, batch_size: int, data_dir: str = "~/data/cifar100", transform=None, ratio: float=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_classes = 100
        self.ratio = ratio
    
    def prepare_data(self):
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            train_ratio = int(len(cifar_full) * (1 - self.ratio))
            val_ratio  = len(cifar_full) - train_ratio
            self.cifar_train, self.cifar_val = random_split(cifar_full, [train_ratio, val_ratio])

        if stage == "test" or stage is None:
            self.cifar_test = CIFAR100(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)

class LitCIFAR10DataModule(LightningDataModule):
    def __init__(self, normalization: str = "imagenet", data_dir: str = "~/data/cifar10"):
        super(LitCIFAR10DataModule, self).__init__()
        self.normalization = normalization 
        self.data_dir=data_dir
        self.mean = self.set_normalization()["mean"]
        self.std = self.set_normalization()["std"]

    def set_normalization(self):
        # Image classification on the CIFAR10 dataset - Albumentations Documentation https://albumentations.ai/docs/autoalbument/examples/cifar10/
        if self.normalization == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif self.normalization == "imagenet":
            # ImageNet - torchbench Docs https://paperswithcode.github.io/torchbench/imagenet/
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        return {"mean": mean, "std": std}

    def get_dataset(self, train, transform):
        return CIFAR10(root=self.data_dir, train=train, transform=transform, download=train)

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.get_dataset(train=True, transform=transform)
        return DataLoader(
            dataset,
            batch_size=128,
            num_workers=2,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.get_dataset(train=False, transform=transform)
        return DataLoader(
            dataset,
            batch_size=128,
            num_workers=2,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()    
    
class SVHNDataModule(LightningDataModule):
    name: str = "svhn"
    
    def __init__(self, batch_size: int, data_dir: str = "~/data/svhn", transform = None, ratio: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_classes = 10
        self.ratio = ratio
    
    def prepare_data(self):
        SVHN(self.data_dir, train=True, download=True)
        SVHN(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            svhn_full = SVHN(self.data_dir, train=True, transform=self.transform)
            train_ratio = int(len(svhn_full) * (1 - self.ratio))
            val_ratio  = len(svhn_full) - train_ratio
            self.svhn_train, self.svhn_val = random_split(svhn_full, [train_ratio, val_ratio])

        if stage == "test" or stage is None:
            self.svhn_test = SVHN(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.svhn_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.svhn_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.svhn_test, batch_size=self.batch_size)