import os
import torch
import numpy as np
from torch import nn
from typing import Optional
from dataclasses import dataclass

from trainer import state
from trainer.callbacks import CallbackRunner
from trainer.config import Config
from trainer.utils import AverageMeter
from trainer.progress import Progress

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class Trainer:

    model: nn.Module
    config: Optional[Config] = None
    train_dataset = None
    valid_dataset = None
    optimizer = None
    scheduler = None
    scaler = None
    fp16 = False

    train_batch_size: Optional[int] = 0
    valid_batch_size: Optional[int] = 0
    current_epoch: Optional[int] = 0
    num_train_steps: Optional[int] = None
    num_valid_steps: Optional[int] = None

    _train_step = 0
    _valid_step = 0
    _test_step = 0

    train_meter = None
    valid_meter = None

    # metrics
    metrics = {}
    metrics['train'] = {}
    metrics['valid'] = {}

    def _init_trainer(
        self,
        train_dataset=None,
        valid_dataset=None,
        config: Config = None,
        **kwargs
    ):
        self.config = config
        self.train_batch_size = config.training_batch_size
        self.valid_batch_size = config.validation_batch_size
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_loader = kwargs.get('train_loader', None)
        self.valid_loader = kwargs.get('valid_loader', None)
        self.train_sampler = kwargs.get('train_sampler', None)
        self.valid_sampler = kwargs.get('valid_sampler', None)
        self.train_collate_fn = kwargs.get('train_collate_fn', None)
        self.valid_collate_fn = kwargs.get('valid_collate_fn', None)

        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.config.training_batch_size,
                num_workers=self.config.num_workers,
                sampler=self.train_sampler,
                shuffle=self.config.train_shuffle,
                collate_fn=self.train_collate_fn,
                drop_last=self.config.train_drop_last,
                pin_memory=self.config.pin_memory,
            )

        if self.valid_loader is None:
            if self.valid_dataset is not None:
                self.valid_loader = torch.utils.data.DataLoader(
                    dataset=valid_dataset,
                    batch_size=self.config.validation_batch_size,
                    num_workers=self.config.num_workers,
                    sampler=self.valid_sampler,
                    shuffle=self.config.valid_shuffle,
                    collate_fn=self.valid_collate_fn,
                    drop_last=self.config.valid_drop_last,
                    pin_memory=self.config.pin_memory,
                )

        self.optimizer, self.scheduler = self.model.fetch_optimizer_and_scheduler()
        if self.optimizer is None:
            raise Exception('optimizer not found')

        self.num_train_steps = int(len(self.train_loader) * self.config.epochs)
        self.num_valid_steps = len(
            self.valid_loader) if self.valid_dataset else None
        self._progress = Progress(self.num_train_steps, self.num_valid_steps)

        if 'callbacks' in kwargs:
            self.callbacks = [self._progress] + kwargs['callbacks']
        else:
            self.callbacks = [self._progress]

        self._callback_runner = CallbackRunner(self.callbacks, self)
        self.train_state = state.TrainingState.TRAIN_START

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, value):
        self._model_state = value

    @property
    def train_state(self):
        return self._train_state

    @train_state.setter
    def train_state(self, value):
        self._train_state = value
        if self._callback_runner is not None:
            self._callback_runner(value)

    def name_to_metric(self, metric_name):
        if metric_name == 'current_epoch':
            return self.current_epoch
        v1 = metric_name.split('_')[0]
        v2 = '_'.join(metric_name.split('_')[1:])

        return self.metrics[v1][v2]

    def update_metrics(self, losses, monitor):
        if self._model_state == state.ModelState.END:
            return

        self.metrics[self._model_state.value].update(monitor)
        self.metrics[self._model_state.value]['loss'] = losses.avg

    def save(self, cur_epoch: Optional[int], path: Optional[str]):
        if os.path.exists(self.config.save_dir) is False:
            os.makedirs(self.config.save_dir)

        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        optimizer_dict = None
        if self.optimizer is not None:
            optimizer_dict = self.optimizer.state_dict()

        scheduler_dict = None
        if self.scheduler is not None:
            scheduler_dict = self.scheduler.state_dict()

        checkpoint = {}
        checkpoint['current_epoch'] = cur_epoch
        checkpoint['model'] = model_state_dict
        checkpoint['optimizer'] = optimizer_dict
        checkpoint['scheduler'] = scheduler_dict
        checkpoint['config'] = self.config

        torch.save(checkpoint, path)

    def load(self, path: Optional[str], config: Config = None):
        self.config = config
        if config is None:
            self.config = Config()

        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def model_fn(self, data):
        for k, v in data.items():
            data[k] = v.to(self.config.device)

        output, loss, metrics = self.model(**data)
        metrics = {k: v.mean() for k, v in metrics.items()}

        return output, loss, metrics

    def _update_monitor(self, losses, metrics):
        monitor = {}

        if self._model_state == state.ModelState.TRAIN:
            metrics_meter = self.train_meter
            # _bs = self.train_loader_bs
            batch_size = self.train_batch_size

        elif self._model_state == state.ModelState.VALID:
            metrics_meter = self.valid_meter
            # _bs = self.valid_loader_bs
            batch_size = self.valid_batch_size
        else:
            raise ValueError('invalid model state')

        for m_m in metrics_meter:
            metrics_meter[m_m].update(metrics[m_m].cpu().detach().numpy(),
                                      batch_size)
            monitor[m_m] = metrics_meter[m_m].avg

        if self._model_state == state.ModelState.TRAIN:
            self.train_meter = metrics_meter
        elif self._model_state == state.ModelState.VALID:
            self.valid_meter = metrics_meter
        else:
            raise ValueError('invalid model state')

        self.update_metrics(losses=losses, monitor=monitor)

        return monitor

    def _update_loss_metrics(self, losses, loss, metrics):
        if self._model_state == state.ModelState.TRAIN:
            if self.train_batch_index == 0:
                self.train_meter = {k: AverageMeter() for k in metrics}
            losses.update(loss.item(), self.train_batch_size)

        elif self._model_state == state.ModelState.VALID:
            if self.valid_batch_index == 0:
                self.valid_meter = {k: AverageMeter() for k in metrics}
            losses.update(loss.item(), self.valid_batch_size)
        else:
            raise ValueError('invalid state')

        monitor = self._update_monitor(losses, metrics)
        if self._model_state == state.ModelState.TRAIN:
            self._train_step += 1
        elif self._model_state == state.ModelState.VALID:
            self._valid_step += 1
        else:
            raise ValueError('invalid state')

        return losses, monitor

    def _set_validation_epoch_start(self, data_loader):
        self.model.eval()
        self.model_state = state.ModelState.VALID
        self.train_state = state.TrainingState.VALID_EPOCH_START
        try:
            self.valid_loader_bs = data_loader.batch_sampler.batch_size
        except:
            self.valid_loader_bs = data_loader._batch_sampler.batch_size

    def _set_validation_epoch_end(self, losses, monitor):
        self.update_metrics(losses=losses, monitor=monitor)
        self.train_state = state.TrainingState.VALID_EPOCH_END

    def valid_step(self, data):
        output, loss, metrics = self.model_fn(data)
        metrics = {k: v.mean() for k, v in metrics.items()}

        return loss, metrics

    def validate(self, data_loader):
        self._set_validation_epoch_start(data_loader)
        losses = AverageMeter()
        for batch_index, data in enumerate(data_loader):
            self.train_state = state.TrainingState.VALID_STEP_START
            self.valid_batch_index = batch_index
            with torch.no_grad():
                loss, metrics = self.valid_step(data)
            self.train_state = state.TrainingState.VALID_STEP_END
            losses, monitor = self._update_loss_metrics(losses, loss, metrics)
        self._set_validation_epoch_end(losses, monitor)

    def process_output(self, output):
        output = output.cpu().detach().numpy()

        return output

    def _set_training_epoch_start(self, data_loader):
        self.model.train()
        self.model_state = state.ModelState.TRAIN
        self.train_state = state.TrainingState.TRAIN_EPOCH_START

    def _set_training_epoch_end(self, losses, monitor):
        self.update_metrics(losses=losses, monitor=monitor)
        self.train_state = state.TrainingState.TRAIN_EPOCH_END

    def train_step(self, data):
        output, loss, metrics = self.model_fn(data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, metrics

    def train(self, data_loader):
        self._set_training_epoch_start(data_loader)
        losses = AverageMeter()
        for batch_index, data in enumerate(data_loader):
            self.train_state = state.TrainingState.TRAIN_STEP_START
            self.train_batch_index = batch_index
            loss, metrics = self.train_step(data)
            losses, monitor = self._update_loss_metrics(losses, loss, metrics)
            self.train_state = state.TrainingState.TRAIN_STEP_END
        self._set_training_epoch_end(losses, monitor)

    def _update_scheduler(self):
        if self.scheduler is not None:
            if self.config.step_scheduler_metric is None:
                self.scheduler.step()
            else:
                step_metric = self.name_to_metric(
                    self.config.step_scheduler_metric)
                self.scheduler.step(step_metric)

    def fit(self, train_dataset, valid_dataset=None, config: Config = None, **kwargs):
        if config is None:
            config = Config()

        self._init_trainer(train_dataset, valid_dataset, config, **kwargs)
        self.train_state = state.TrainingState.TRAIN_START
        best_loss = np.inf
        for epoch in range(self.config.epochs):
            self.train_state = state.TrainingState.EPOCH_START
            self.train(self.train_loader)
            if self.valid_loader is not None:
                self.validate(self.valid_loader)
            self._update_scheduler()
            self.train_state = state.TrainingState.EPOCH_END
            self.current_epoch += 1

            if best_loss > self.metrics['valid']['loss']:
                best_loss = self.metrics['valid']['loss']
                self.save(
                    epoch,
                    f'{self.config.save_dir}/{self.config.model_name}_best.pth')
            else:
                self.save(
                    epoch,
                    f'{self.config.save_dir}/{self.config.model_name}_last.pth')

        self.train_state = state.TrainingState.TRAIN_END

    def predict(self):
        pass
