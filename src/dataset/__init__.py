from pytorch_lightning import LightningDataModule
from dataset.datamodule import CIFAR10DataModule, SVHNDataModule, CIFAR100DataModule


def get_dataset(dataset_name: str, batch_size: int) -> LightningDataModule:
    match dataset_name:
        case "cifar10":
            return CIFAR10DataModule(batch_size)
        case "svhn":
            return SVHNDataModule(batch_size)
        case "cifar100":
            return CIFAR100DataModule(batch_size)
    raise ValueError("dataset_name must be `cifar10`, `cifar100`, `svhn`")