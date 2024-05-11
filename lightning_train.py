import argparse
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from dataset.datamodule import CIFAR10DataModule, CIFAR100DataModule, SVHNDataModule
from trainer.torch_lightning.image_classifier import ImageClassifierTrainer, VAL_LOSS

from mlp.cnn import CNN
from utils import set_randomseed


if __name__ == "__main__":
    set_randomseed()

    p = argparse.ArgumentParser(description="CNN")
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10", "cifar100", "svhn"])
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--warmup_epochs", default=10, type=int)
    args = p.parse_args()

    datamodule = CIFAR100DataModule(args.batch_size)
    match args.dataset:
        case "cifar10":
            datamodule = CIFAR10DataModule(args.batch_size)
        case "svhn":
            datamodule = CIFAR10DataModule(args.batch_size)
        case "cifar100":
            datamodule = CIFAR100DataModule(args.batch_size)

    logger = TensorBoardLogger("logs", name="cnn")

    checkpoint = ModelCheckpoint(monitor=VAL_LOSS, mode="min", save_last=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        logger=logger,
        accelerator="auto",
        deterministic=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        max_epochs=100,
        precision=32,
        callbacks=[checkpoint, lr_monitor],
    )

    cnn = CNN(datamodule.num_classes)
    model = ImageClassifierTrainer(
        args.epochs, args.warmup_epochs, args.lr, datamodule.num_classes, cnn, trainer, criterion
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule.test_dataloader())
