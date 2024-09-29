import torch
import torch.nn.functional as F
from torch import nn
from pytorch_lightning import LightningModule, Trainer
from typing import List, Dict, Any, Tuple
from torchmetrics import MeanMetric
from trainer.lightning.const import *


class MAETrainer(LightningModule):

    def __init__(
        self,
        config: Dict,
        model: nn.Module = None,
        trainer: Trainer = None,
        criterion: nn.Module = None,
        *args,
        **kwargs,
    ):
        super(MAETrainer, self).__init__(*args, **kwargs)

        self.save_hyperparameters(config)  # save the `hparams` to the `self` object

        self.criterion = criterion
        self.model = model
        self.trainer = trainer
        # ---------- Metrics ----------
        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()

        self._init_weights()

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        images = batch["image"]
        pred_img, mask = self.model(images)
        loss = torch.mean((pred_img - images) ** 2 * mask) / self.model.mask_ratio
        metrics = self.monitor_metrics(pred_img, images)
        return loss, metrics

    def training_step(self, batch: Dict, _batch_idx: int) -> torch.Tensor:
        loss, metrics = self(batch)  # self.forward(batch)
        self.train_loss(loss)
        self.log(TRAIN_LOSS, loss, on_step=True, on_epoch=False, logger=True)
        return loss

    def validation_step(self, batch: Dict, _batch_idx: int) -> torch.Tensor:
        loss, metrics = self(batch)  # self.forward(batch)
        self.valid_loss(loss)
        self.log(VAL_LOSS, loss)
        return loss

    def test_step(self, batch: List[torch.Tensor], _batch_idx: int) -> None:
        pass

    def setup_steps(self, stage: Any = None) -> int:
        # After updating to 1.5.2, NotImplementedError: `train_dataloader` · Discussion #10652 · PyTorchLightning/pytorch-lightning https://github.com/PyTorchLightning/pytorch-lightning/discussions/10652
        train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
        return len(train_loader)

    def configure_optimizers(self) -> Dict:
        optimizer: torch.optim.Optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.warmup_epochs / self.hparams.epochs,
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

    def monitor_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict:
        pass

    def optimizer_zero_grad(
        self, _epoch: int, _batch_idx: int, optimizer: torch.optim.Optimizer, _optimizer_idx: int
    ) -> None:
        optimizer.zero_grad(set_to_none=True)

    def _init_weights(self, pretrained: str = None) -> None:
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if n.startswith("head"):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
