import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    experiment_name = "default"
    device: Optional[str] = "cuda"  # cuda or cpu
    model_name: Optional[str] = "cnn"
    epochs: Optional[int] = 100
    training_batch_size: Optional[int] = 16
    validation_batch_size: Optional[int] = 16
    test_batch_size: Optional[int] = 16
    gradient_accumlation_steps: Optional[int] = 1
    fp: Optional[int] = 32  # 16 or 32
    num_workers: Optional[int] = 2
    pin_memory: Optional[bool] = True
    save_dir: Optional[str] = "save"
    lr: Optional[float] = 1e-3
    dataset: Optional[str] = "cifar10"  # cifar10, cifar100, svhn
    save_top_k: Optional[int] = 3
    patience: Optional[int] = 4

    # logging
    dirpath: Optional[str] = "logs"

    # scheduler parameters
    step_scheduler_after: Optional[str] = "epoch"  # "epoch" or "batch"
    step_scheduler_metric: Optional[str] = None

    train_shuffle: Optional[bool] = True
    valid_shuffle: Optional[bool] = False
    test_shuffle: Optional[bool] = False
    train_drop_last: Optional[bool] = False
    valid_drop_last: Optional[bool] = False
    test_drop_last: Optional[bool] = False
    pin_memory: Optional[bool] = True

    # classification
    num_classes: Optional[int] = 10  # cifar10 or cifar100
