###=============================================================================
### Imports
###=============================================================================

import math
import operator
from os import path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trainutils import utils
from trainutils.datasets.utils import IndexedDataset
from trainutils.models import save_model
from trainutils.evaluation import eval_model
from trainutils.trainer import Trainer

###=============================================================================
### Implementation
###=============================================================================

###-----------------------------------------------------------------------------
### Base
###-----------------------------------------------------------------------------

class Callback():
    def on_fit_begin(self, trainer: Trainer, args: Dict) -> None:
        pass

    def on_fit_end(self, trainer: Trainer, args: Dict) -> None:
        pass

    def on_epoch_begin(self, trainer: Trainer, args: Dict) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer, args: Dict) -> None:
        pass

    def on_batch_begin(self, trainer: Trainer, args: Dict) -> None:
        pass

    def on_batch_end(self, trainer: Trainer, args: Dict) -> None:
        pass

###-----------------------------------------------------------------------------
### Callbacks
###-----------------------------------------------------------------------------

class PrintLogs(Callback):
    def on_epoch_end(self, trainer: Trainer, args: Dict) -> None:
        n_metrics = len(trainer.logs.keys())
        for i, (metric, values) in enumerate(trainer.logs.items()):
            end = '\n' if i % 2 != 0 or i == n_metrics - 1 else '\t'
            print(f'{metric: <20} {values[-1]:.4f}', end=end)


class SaveLogs(Callback):
    def on_epoch_end(self, trainer: Trainer, args: Dict) -> None:
        fpath = path.join(trainer.save_dir, '_logs_training.csv')
        utils.save_dict(fpath, trainer.logs)


class SaveCheckpoints(Callback):
    def __init__(
        self,
        save_freq: int,
        monitor: str = 'val_accuracy',
        maximize: bool = True,
        start: int = 0
    ) -> None:
        self.save_freq = save_freq
        self.monitor = monitor
        self.best_metric = -math.inf if maximize else math.inf
        self.compare = operator.gt if maximize else operator.lt
        self.start = start

    def on_epoch_end(self, trainer: Trainer, args: Dict) -> None:
        epoch = args['epoch']

        # save model frequently
        if (
            epoch == self.start or
            epoch == trainer.n_epochs or
            (epoch + self.start) % self.save_freq == 0
        ):
            fpath = path.join(trainer.save_dir, f'model_epoch_{epoch}.pth.tar')
            save_model(fpath, trainer.model, epoch)

        # save best model
        metric = trainer.logs[self.monitor][-1]
        if self.compare(metric, self.best_metric):
            self.best_metric = metric
            fpath = path.join(trainer.save_dir, 'model_best.pth.tar')
            save_model(fpath, trainer.model, epoch)


class SaveTrainOutputs(Callback):
    def __init__(self):
        self.outputs = list()

    def on_batch_end(self, trainer: Trainer, args: Dict) -> None:
        loader = args['loader']
        assert type(loader.dataset) is IndexedDataset

        idxs = args['rest'][0]
        output = args['output']
        softmax_output = F.softmax(output, dim=1)
        for i, o in zip(idxs, softmax_output):
            i, o = i.item(), o.tolist()
            self.outputs.append((i, o))

    def on_epoch_end(self, trainer: Trainer, args: Dict) -> None:
        ordered_outputs = [o for _, o in sorted(self.outputs)]

        epoch = args['epoch']
        fpath = path.join(trainer.save_dir, f'_model_outputs_epoch_{epoch}.npy')
        np.save(fpath, ordered_outputs)
        self.outputs = list()


class SaveModelOutputs(Callback):
    def __init__(self, loader):
        assert type(loader.dateset) is IndexedDataset
        self.loader = loader

    def on_epoch_end(self, trainer: Trainer, args: Dict) -> None:
        trainer.model.eval()

        outputs = list()
        for data, _, idxs in self.loader:
            data = data.to(trainer.device)
            with torch.no_grad():
                output = trainer.model(data)
                softmax_output = F.softmax(output, dim=1)
                for i, o in zip(idxs, softmax_output):
                    i, o = i.item(), o.tolist()
                    outputs.append((i, o))

        ordered_outputs = [o for _, o in sorted(outputs)]

        epoch = args['epoch']
        fpath = path.join(trainer.save_dir, f'_model_outputs_epoch_{epoch}.npy')
        np.save(fpath, ordered_outputs)


class EarlyStop(Callback):
    def __init__(
        self,
        warmup: int = 10,
        patience: int = 20,
        monitor: str = 'val_accuracy',
        maximize: bool = True
    ) -> None:
        self.warmup = warmup
        self.patience = patience
        self.monitor = monitor
        self.wait = 0
        self.best_metric = -math.inf if maximize else math.inf
        self.compare = operator.gt if maximize else operator.lt

    def on_epoch_end(self, trainer: Trainer, args: Dict) -> None:
        epoch = args['epoch']
        if epoch > self.warmup:
            metric = trainer.logs[self.monitor][-1]
            if self.compare(metric, self.best_metric):
                self.best_metric = metric
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    trainer.cancel_train = True
                    print(f'Training stopped at epoch {epoch}...')


class Evaluate(Callback):
    def __init__(self, loader: DataLoader, prefix: str) -> None:
        self.loader = loader
        self.prefix = prefix

    def on_epoch_end(self, trainer: Trainer, args: Dict) -> None:
        results = eval_model(trainer.model, self.loader, trainer.criterion)
        trainer.logs[f'{self.prefix}_loss'].append(results['loss'])
        trainer.logs[f'{self.prefix}_accuracy'].append(results['accuracy'])
