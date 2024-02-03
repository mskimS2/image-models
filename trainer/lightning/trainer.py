import torch
import pytorch_lightning as pl
from torch import nn
from typing import Any, Dict, List
from torchmetrics import MeanMetric


class ImageClassifierTrainer(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        super(ImageClassifierTrainer, self).__init__()
        self.save_hyperparameters(config)
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: List[torch.Tensor], _batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_loss(loss)
        self.log(
            name="train/loss", 
            value=self.train_loss, 
            on_step=True, 
            on_epoch=False, 
            logger=True,
        )
        return loss
    
    def validation_step(self, batch: List[torch.Tensor], _batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.valid_loss(loss)
        self.log(
            name="valid/loss",
            value=self.valid_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }