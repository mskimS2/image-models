import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torch.utils.data import random_split, DataLoader


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: str = "./", transform = None, ratio: float = 0.2):
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
    
    
class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: str = "./", transform=None, ratio: float=0.2):
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
    
    
class SVHNDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: str = "./", transform = None, ratio: float = 0.2):
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