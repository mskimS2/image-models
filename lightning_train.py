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
from dataset.datamodule import LitCIFAR10DataModule
from trainer.torch_lightning.image_classifier import ImageClassifierTrainer, VAL_LOSS

from mlp.cnn import CNN
from configs.config import Config
from utils import set_randomseed


if __name__ == "__main__":
    set_randomseed()
    logger = TensorBoardLogger("logs", name="cnn")

    checkpoint = ModelCheckpoint(monitor=VAL_LOSS, mode="min",save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

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

    config = Config(
        training_batch_size=128,
        validation_batch_size=128,
        epochs=300,
        lr=1e-3,
        num_classes=10,
        step_scheduler_after='epoch',
        step_scheduler_metric='valid_f1',
        model_name='mlpmixer',
        num_workers=2,
    )
    datamodule = LitCIFAR10DataModule()
    model = ImageClassifierTrainer(100, 10, 1e-3, 10, CNN(), trainer, nn.CrossEntropyLoss())
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule.test_dataloader())