import torch
from torch import nn
from typing import List, Dict, Any
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric


TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"

class ImageClassifier(LightningModule):
    hparams: Dict[Any]
    
    def __init__(
        self, 
        model: nn.Module, 
        loss_fn: nn.Module, 
        config: Dict[Any],
    ):
        super(ImageClassifier, self).__init__()
        
        self.save_hyperparameters(config) # save the `hparams` to the `self` object
        self.model = model
        self.loss_fn = loss_fn
        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)
    
    def training_step(self, batch: List[torch.Tensor], _batch_idx: int) -> torch.Tensor:
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        self.train_loss(loss)
        self.log(TRAIN_LOSS, value=self.train_loss, on_step=True, on_epoch=False, logger=True)
        return loss
    
    def validation_step(self, batch: List[torch.Tensor], _batch_idx: int) -> None:
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        self.valid_loss(loss)
        self.log(VAL_LOSS, value=self.valid_loss, on_step=False, on_epoch=True, logger=True)
    
    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr,
        )
        
        scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.warmpup_epochs / self.hparams.epochs,
            anneal_strategy="linear",
            div_factor=self.hparams.lr / self.hparams.min_lr,
            final_div_factor=1e6,   
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def optimizer_zero_grad(
        self, 
        _epoch: int, 
        _batch_idx: int, 
        optimizer: torch.optim.Optimizer, 
        _optimizer_idx: int,
    ):
        optimizer.zero_grad(set_to_none=True)