import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.custom import Cifar10, Cifar100, SVHN


class ImageDatasetModule(LightningDataModule):
    trainer: Trainer

    def __init__(self, config, dataset) -> None:
        super(ImageDatasetModule, self).__init__()

        self.save_hyperparameters(config)
        self.dataset = dataset

    def get_dataset(self, _stage: str = None):
        self.dataset(split="train")
        self.dataset(split="validation")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset(transforms, split="train"),
            batch_size=self.hparams.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=0 if self.on_device else self.trainer.num_devices * 4,
            prefetch_factor=2,
            persistant_workers=True,
            pin_memory=not self.on_device,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset(transforms, split="validation"),
            batch_size=self.hparams.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=0 if self.on_device else self.trainer.num_devices * 4,
            prefetch_factor=2,
            persistant_workers=True,
            pin_memory=not self.on_device,
        )

    @property
    def on_device(self) -> bool:
        return isinstance(self.trainer.accelerator, CPUAccelerator)


class CIFAR10DataModule(LightningDataModule):
    name: str = "cifar10"

    def __init__(self, batch_size: int, data_dir: str = "./data/cifar10"):
        super(CIFAR10DataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = 10

    def get_dataset(self, train, transform):
        return Cifar10(root=self.data_dir, train=train, transform=transform, download=train)

    def train_dataloader(self):
        transform = A.Compose(
            [
                A.Resize(32, 32),
                A.HorizontalFlip(p=0.5),
                # ImageNet - torchbench Docs https://paperswithcode.github.io/torchbench/imagenet/
                A.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        dataset = self.get_dataset(train=True, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        transform = A.Compose(
            [
                A.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        dataset = self.get_dataset(train=False, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=2,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()


class CIFAR100DataModule(LightningDataModule):
    name: str = "cifar100"

    def __init__(self, batch_size: int, data_dir: str = "./data/cifar100"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = 100

    def get_dataset(self, train, transform):
        return Cifar100(root=self.data_dir, train=train, transform=transform, download=train)

    def train_dataloader(self):
        transform = A.Compose(
            [
                A.Resize(32, 32),
                A.HorizontalFlip(p=0.5),
                # ImageNet - torchbench Docs https://paperswithcode.github.io/torchbench/imagenet/
                A.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        dataset = self.get_dataset(train=True, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        transform = A.Compose(
            [
                A.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        dataset = self.get_dataset(train=False, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()


class SVHNDataModule(LightningDataModule):
    name: str = "svhn"

    def __init__(self, batch_size: int, data_dir: str = "./data/svhn"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = 10

    def get_dataset(self, split: str, transform):
        return SVHN(root=self.data_dir, split=split, transform=transform, download=True)

    def train_dataloader(self):
        transform = A.Compose(
            [
                A.Resize(32, 32),
                A.HorizontalFlip(p=0.5),
                # ImageNet - torchbench Docs https://paperswithcode.github.io/torchbench/imagenet/
                A.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        dataset = self.get_dataset(split="train", transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        transform = A.Compose(
            [
                A.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        dataset = self.get_dataset(split="test", transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()
