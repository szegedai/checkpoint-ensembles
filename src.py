###=============================================================================
### Import
###=============================================================================

import copy
import os
from heapq import heappush, heappushpop
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from trainutils import utils
from trainutils.datasets.utils import IndexedDataset
from trainutils.trainer import Callback, Trainer

###=============================================================================
### Implementation
###=============================================================================

class LogTopModels(Callback):
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.top_k_storage = list()

    def on_epoch_end(self, trainer: Trainer, args: Dict) -> None:
        epoch = args['epoch']
        self._store_top_k(trainer.model, trainer.logs, epoch)

    def on_fit_end(self, trainer: Trainer, args: Dict) -> None:
        self._save_top_k(trainer.save_dir)

    def _store_top_k(self, model, logs: Dict, epoch: int) -> None:
        current_loss = logs['val_loss'][-1]

        if len(self.top_k_storage) < self.top_k:
            model_clone = copy.deepcopy(model).to('cpu')
            item = (-current_loss, -epoch, model_clone)
            heappush(self.top_k_storage, item)
        elif current_loss < -self.top_k_storage[0][0]:
            model_clone = copy.deepcopy(model).to('cpu')
            item = (-current_loss, -epoch, model_clone)
            heappushpop(self.top_k_storage, item)

    def _save_top_k(self, save_dir: str) -> None:
        # save models
        fpath = os.path.join(save_dir, 'models.pkl')
        state = [{'epoch': -e, 'loss': -l, 'model': m.state_dict()}
                  for l, e, m in sorted(self.top_k_storage, reverse=True)]
        utils.serialize(fpath, state)

        # save epochs
        fpath = os.path.join(save_dir, 'epochs.npy')
        epochs = [d['epoch'] for d in state]
        np.save(fpath, epochs)


class AggregatePredictions(Callback):
    def __init__(self, loader, start_epoch, end_epoch):
        assert type(loader.dataset) is IndexedDataset
        self.loader = loader

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

        self.logits = dict()
        self.predictions = dict()

    def on_epoch_end(self, trainer, args):
        if self.start_epoch <= args['epoch'] < self.end_epoch:
            device = trainer.device
            model = trainer.model

            model.eval()
            for data, target, idxs in self.loader:
                data, target = data.to(device), target.to(device)
                idxs = idxs.tolist()

                with torch.no_grad():
                    output = model(data)
                    softmax_output = F.softmax(output, dim=1).detach().cpu()

                for i, o, t in zip(idxs, softmax_output, target):
                    # ensemble
                    if i not in self.logits:
                        self.logits[i] = torch.zeros(o.shape)
                    self.logits[i] += o

                    # majority vote
                    if i not in self.predictions:
                        self.predictions[i] = list()
                    self.predictions[i].append(torch.argmax(o).ne(t).item())

    def on_fit_end(self, trainer, args):
        targets = torch.tensor(self.loader.dataset.dataset.targets)
        save_dir = trainer.save_dir
        epoch_range = f'{self.start_epoch}-{self.end_epoch}'

        # ensemble
        ensemble_resuls = torch.zeros(len(targets), dtype=torch.int32)
        for k, v in self.logits.items():
            ensemble_resuls[k] = torch.argmax(v)
        ensemble_resuls = torch.ne(targets, ensemble_resuls).tolist()
        np.save(f'{save_dir}/ensemble_results_{epoch_range}.npy',
                ensemble_resuls)

        # majority vote
        mvote_results = torch.zeros(len(targets), dtype=torch.int32)
        for k, v in self.predictions.items():
            mvote_results[k] = sum(v) >= len(v)//2
        np.save(f'{save_dir}/majority_vote_results_{epoch_range}.npy',
                mvote_results)
