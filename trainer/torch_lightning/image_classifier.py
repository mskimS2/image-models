import torch
import torch.nn.functional as F
from torch import nn
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from typing import List, Dict, Any, Tuple
from torchmetrics import MeanMetric, Accuracy, F1Score, Recall, Precision


TRAIN_LOSS = "train/loss"
TRAIN_ACC = "train/acc"
TRAIN_F1 = "train/f1"
TRAIN_RECALL = "train/recall"
TRAIN_PRECISION = "train/precision"
VAL_LOSS = "val/loss"
VAL_ACC = "val/acc"
VAL_F1 = "val/f1"
VAL_RECALL = "val/recall"
VAL_PRECISION = "val/precision"
TEST_LOSS = "test/loss"
TEST_ACC = "test/acc"
TEST_F1 = "test/f1"
TEST_RECALL = "test/recall"
TEST_PRECISION = "test/precision"


class ImageClassifierTrainer(LightningModule):

    def __init__(
        self,
        config: Dict,
        num_classes: int = None,
        model: nn.Module = None,
        trainer: Trainer = None,
        criterion: nn.Module = None,
        *args,
        **kwargs,
    ):
        super(ImageClassifierTrainer, self).__init__(*args, **kwargs)

        self.save_hyperparameters(config)  # save the `hparams` to the `self` object

        self.criterion = criterion
        self.model = model
        self.trainer = trainer
        # ---------- Metrics ----------
        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1score_metric = F1Score(task="multiclass", num_classes=num_classes)
        self.recall_metric = Recall(task="multiclass", average="macro", num_classes=num_classes)
        self.precision_metric = Precision(task="multiclass", average="macro", num_classes=num_classes)

        self._init_weights()

    def forward(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        images, labels = batch
        preds = self.model(images)
        loss = self.criterion(preds, labels)
        metrics = self.monitor_metrics(preds, labels)
        return loss, metrics

    def training_step(self, batch: List[torch.Tensor], _batch_idx: int) -> torch.Tensor:
        loss, metrics = self(batch)  # self.forward(batch)
        self.train_loss(loss)
        self.log(TRAIN_LOSS, loss, on_step=True, on_epoch=False, logger=True)
        self.log(TRAIN_ACC, metrics.get("acc"), on_step=True, on_epoch=False, logger=True)
        self.log(TRAIN_F1, metrics.get("f1"), on_step=True, on_epoch=False, logger=True)
        self.log(TRAIN_RECALL, metrics.get("recall"), on_step=True, on_epoch=False, logger=True)
        self.log(TRAIN_PRECISION, metrics.get("precision"), on_step=True, on_epoch=False, logger=True)
        return loss

    def validation_step(self, batch: List[torch.Tensor], _batch_idx: int) -> torch.Tensor:
        loss, metrics = self(batch)  # self.forward(batch)
        self.valid_loss(loss)
        self.log(VAL_LOSS, loss)
        self.log(VAL_ACC, metrics.get("acc"), on_epoch=True, logger=True)
        self.log(VAL_F1, metrics.get("f1"), on_epoch=True, logger=True)
        self.log(VAL_RECALL, metrics.get("recall"), on_epoch=True, logger=True)
        self.log(VAL_PRECISION, metrics.get("precision"), on_epoch=True, logger=True)
        return loss

    def test_step(self, batch: List[torch.Tensor], _batch_idx: int) -> None:
        loss, metrics = self.forward(batch)
        self.log(TEST_ACC, metrics.get("acc"), logger=True)
        self.log(TEST_F1, metrics.get("f1"), logger=True)
        self.log(TEST_RECALL, metrics.get("recall"), logger=True)
        self.log(TEST_PRECISION, metrics.get("precision"), logger=True)

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

    def monitor_metrics(self, outputs, targets):
        return {
            "acc": self.accuracy_metric(targets, torch.argmax(outputs, dim=-1)),
            "f1": self.f1score_metric(targets, torch.argmax(outputs, dim=-1)),
            "recall": self.recall_metric(targets, torch.argmax(outputs, dim=-1)),
            "precision": self.precision_metric(targets, torch.argmax(outputs, dim=-1)),
        }

    def optimizer_zero_grad(self, _epoch: int, _batch_idx: int, optimizer: torch.optim.Optimizer, _optimizer_idx: int):
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
