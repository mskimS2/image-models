import torch
import numpy as np
from typing import List
from torch import nn
from sklearn import metrics
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers import PytorchProfiler

from dataset.datamodule import ImageDataModule
from dataset.custom import CIFAR10
from trainer.custom.config import Config
from mlp.mlp_mixer import MlpMixer
from transform.auto_aug import AutoAugment
from trainer.lightning import ImageClassifier
from trainer.lightning import VAL_LOSS


if __name__ == "__main__":
    RUN_NAME = "lightning"

    train_aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ]
    )
    train_dataset = CIFAR10(root="./data/",
                            train=True,
                            download=True,
                            transform=train_aug)

    test_aug = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ]
    )
    test_dataset = CIFAR10(root="./data/",
                           train=False,
                           download=True,
                           transform=test_aug)

    config = Config(
        training_batch_size=128,
        validation_batch_size=128,
        epochs=300,
        step_scheduler_after='epoch',
        step_scheduler_metric='valid_f1',
        model_name='mlpmixer',
        num_workers=4,
        save_top_k=3,
        patience=10,
        log_interval=100,
    )

    model = MlpMixer(
        in_channels=3,
        img_size=32,
        patch_size=4,
        tokens_mlp_dim=512,
        channels_mlp_dim=64,
        hidden_dim=128,
        dropout_p=0.,
        n_layers=8,
        n_classes=10,
    )
    
    datamodule = ImageDataModule(config)
    
    loss_fn = nn.CrossEntropyLoss()
    
    logger = WandbLogger(
        name="awesome-mlp", 
        group="mlp-mixer",
        project=RUN_NAME,
        offline=False,
        log_model=True,
    )
    
    logger.watch(model, log="all", log_freq=config.log_interval)
    
    classfier = ImageClassifier(model, loss_fn, config)
    
    callbacks: List[Callback] = [
        ModelCheckpoint(
            dirpath="build/{RUN_NAME}", 
            filename="epoch={epoch:02d}-val_loss={"+ VAL_LOSS +":.2f}",
            monitor=VAL_LOSS,
            save_top_k=config.save_top_k,
            mode="min",
            save_weights_only=False,
            auto_insert_metric_name=False,
            save_last=True,
        ),
        LearningRateMonitor(
            logging_interval="step", # epoch or step
            # log_momentum=True,
        ),
        EarlyStopping(
            monitor=VAL_LOSS,
            patience=config.patience,
            mode="min",
            strict=True,
            check_finite=True,
        ),
        RichModelSummary(max_depth=1),
        RichProgressBar(
            theme=RichProgressBarTheme(
                description="green_yellow",
                progress_bar="green1",
                progress_bar_finished="green1",
                progress_bar_pulse="#6206E0",
                batch_progress="green_yellow",
                time="grey82",
                processing_speed="grey82",
                metrics="grey82",
            ),
        ),
    ]
    
    # profiler = PytorchProfiler(
    #     dirpath="logs/", 
    #     filename="profile-{RUN_NAME}", 
    #     export_to_chrome=True,
    # )
    
    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=config.epochs,
        gradient_clip_algorithm="norm",
        gradient_clip_val=config.gradient_clip,
        log_every_n_steps=config.log_interval,
        track_grad_norm=2,
        accelerator="auto",
        devices="auto", # -1
        # precision=16,
        # profiler=profiler,
    )
    
    trainer.fit(
        model=classfier, 
        datamodule=datamodule,
        config=config,
    )
