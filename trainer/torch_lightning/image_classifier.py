import torch
import torch.nn.functional as F
from torch import nn
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from typing import List, Dict, Any, Tuple
from torchmetrics import MeanMetric, Accuracy


TRAIN_LOSS = "train/loss"
TRAIN_ACC = "train/acc"
VAL_LOSS = "val/loss"
VAL_ACC = "val/acc"
TEST_LOSS = "test/loss"
TEST_ACC = "test/acc"

class ImageClassifierTrainer(LightningModule):
    
    def __init__(
        self, 
        epochs: int = 100,
        warmpup_epochs: int = 10,
        lr: float = 1e-3,
        num_classes: int = 10,
        model: nn.Module = None,
        trainer: Trainer = None,
        criterion: nn.Module = None,
        *args, 
        **kwargs,
    ):
        super(ImageClassifierTrainer, self).__init__(*args, **kwargs)
        
        self.save_hyperparameters() # save the `hparams` to the `self` object

        self.criterion = criterion
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.model = model
        self.trainer = trainer
        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()

    def forward(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        images, labels = batch
        preds = self.model(images)
        loss = self.criterion(preds, labels)
        accuracy = self.accuracy(preds, labels) * 100
        return loss, accuracy 

    def training_step(self, batch: List[torch.Tensor], _batch_idx: int) -> torch.Tensor:
        loss, accuracy = self(batch) # self.forward(batch)
        self.train_loss(loss)
        self.log(TRAIN_LOSS, loss, on_step=True, on_epoch=False, logger=True)
        self.log(TRAIN_ACC, accuracy, on_step=True, on_epoch=False, logger=True)
        return loss

    def validation_step(self, batch: List[torch.Tensor], _batch_idx: int) -> torch.Tensor:
        loss, accuracy = self(batch) # self.forward(batch)
        self.valid_loss(loss)
        self.log(VAL_LOSS, loss)
        self.log(VAL_ACC, accuracy)
        return loss

    def test_step(self, batch: List[torch.Tensor], _batch_idx: int) -> None:
        loss, accuracy = self.forward(batch)
        self.log(TEST_ACC, accuracy)

    def setup_steps(self, stage: Any = None) -> int:
        # After updating to 1.5.2, NotImplementedError: `train_dataloader` · Discussion #10652 · PyTorchLightning/pytorch-lightning https://github.com/PyTorchLightning/pytorch-lightning/discussions/10652
        train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
        return len(train_loader)

    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.warmpup_epochs / self.hparams.epochs,
            anneal_strategy="linear",  
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        
    def optimizer_zero_grad(self, _epoch: int, _batch_idx: int, optimizer: torch.optim.Optimizer, _optimizer_idx: int):
        optimizer.zero_grad(set_to_none=True)