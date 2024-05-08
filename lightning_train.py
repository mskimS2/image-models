import os
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from dataset.datamodule import LitCIFAR10DataModule
from mlp.test_model import LitCIFAR10Model
from utils import set_randomseed


def main() -> None:
    set_randomseed()
    logger = TensorBoardLogger("logs", name="cnn")

    checkpoint = ModelCheckpoint(monitor="acc/val", mode="max",save_last=True)
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

    datamodule = LitCIFAR10DataModule()
    model = LitCIFAR10Model(trainer)

    # if cfg.runs.evaluate:
    #     hparams = OmegaConf.load(cfg.test.hparams)
    #     model = LitCIFAR10Model.load_from_checkpoint(
    #         checkpoint_path=cfg.test.checkpoint, **hparams
    #     )
    #     trainer.test(model, datamodule.test_dataloader())
    # else:
    trainer.fit(model, datamodule)
    # NOTE After changing to pytorch-lightning 1.5.2, omitting the argument of trainer.test() does not work. · Discussion #10747 · PyTorchLightning/pytorch-lightning https://github.com/PyTorchLightning/pytorch-lightning/discussions/10747
    trainer.test(model, datamodule.test_dataloader())


if __name__ == "__main__":
    main()