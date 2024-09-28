import argparse
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from dataset import get_dataset
from trainer.lightning.image_classifier import ImageClassifierTrainer, VAL_LOSS
from configs.config import Config
from models import get_model
from utils import set_randomseed


if __name__ == "__main__":
    set_randomseed()

    p = argparse.ArgumentParser()
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument(
        "--model",
        default="cnn",
        type=str,
        choices=["cnn", "mlpmixer", "resmlp", "convmixer", "vit", "gmlp", "mae", "mae_linear_probing"],
    )
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--dirpath", default="logs", type=str)
    p.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10", "cifar100", "svhn"])
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--warmup_epochs", default=10, type=int)
    p.add_argument("--fp", default=32, type=int, choices=[16, 32])
    p.add_argument("--save_top_k", default=3, type=int)
    p.add_argument("--patience", default=4, type=int)
    p.add_argument("--ssl", default=False, type=bool, help="use classification or pretrain(self-supervised learning)")
    p.add_argument("--pretrained_path", default="", type=str, help="pretrained model(backbone model) path")
    args = p.parse_args()

    datamodule = get_dataset(args.dataset, args.batch_size)

    logger = TensorBoardLogger(args.dirpath, name=args.model)

    criterion = nn.CrossEntropyLoss()
    if args.ssl:
        criterion = nn.MSELoss()

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",  # -1
        deterministic=False,
        precision=args.fp,
        enable_model_summary=False,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=args.dirpath,
                filename="epoch={epoch:03d}-val_loss={" + VAL_LOSS + ":.2f}",
                monitor=VAL_LOSS,
                save_top_k=args.save_top_k,
                mode="min",
                save_weights_only=False,
                auto_insert_metric_name=False,
                save_last=True,
            ),
            EarlyStopping(
                monitor=VAL_LOSS,
                patience=args.patience,
                mode="min",
                strict=True,
                check_finite=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    cfg = Config()
    model = get_model(cfg)
    classifier = ImageClassifierTrainer(args, datamodule.num_classes, model, trainer, criterion)

    trainer.fit(classifier, datamodule)
    if not args.ssl:
        trainer.test(classifier, datamodule.test_dataloader())
