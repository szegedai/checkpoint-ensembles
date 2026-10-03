###=============================================================================
### Imports
###=============================================================================

from typing import Callable, List, Optional

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainutils.evaluation import eval_model
from trainutils.trainer import Trainer

###=============================================================================
### Implementation
###=============================================================================

class ClassifierTrainer(Trainer):
    def __init__(
        self,
        model: Module,
        criterion: Callable,
        optimizer: Optimizer,
        n_epochs: int,
        save_dir: str,
        scheduler: Optional[Callable] = None,
        callbacks: List = list()
    ) -> None:
        super().__init__(model,
                         criterion,
                         optimizer,
                         n_epochs,
                         save_dir,
                         scheduler,
                         callbacks)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Module:
        self._do_callbacks('on_fit_begin', locals())

        # iterate epochs
        for epoch in range(1, self.n_epochs+1):
            self._do_callbacks('on_epoch_begin', locals())

            # train step
            self.train(train_loader, epoch)
            # eval step
            self.eval(val_loader)

            self._do_callbacks('on_epoch_end', locals())

            if self.cancel_train:
                break

        self._do_callbacks('on_fit_end', locals())

        return self.model

    def train(self, loader: DataLoader, epoch: int) -> None:
        self.model.train()

        n_samples = 0
        n_correct = 0
        sum_loss = 0

        # iterate batches
        loader_tqdm = tqdm(loader, desc=f'{epoch}/{self.n_epochs}', ncols=80)
        for data, target, *rest in loader_tqdm:
            data, target = data.to(self.device), target.to(self.device)

            self._do_callbacks('on_batch_begin', locals())

            # train
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # batch results
            sum_loss += loss.item() * len(target)
            n_correct += torch.argmax(output, 1).eq(target).sum().item()
            n_samples += len(target)

            self._do_callbacks('on_batch_end', locals())

        # scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        # epoch results
        model_loss = sum_loss / n_samples
        model_accuracy = n_correct / n_samples
        self.logs['train_loss'].append(model_loss)
        self.logs['train_accuracy'].append(model_accuracy)

    def eval(self, loader: DataLoader) -> None:
        self.model.eval()

        # evaluate
        results = eval_model(self.model, loader, self.criterion)
        self.logs['val_loss'].append(results['loss'])
        self.logs['val_accuracy'].append(results['accuracy'])
