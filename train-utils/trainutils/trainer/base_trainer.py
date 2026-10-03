###=============================================================================
### Imports
###=============================================================================

import os
from collections import defaultdict
from typing import Any, Callable, List, Optional

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

###=============================================================================
### Implementation
###=============================================================================

class Trainer():
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
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.save_dir = save_dir
        self.scheduler = scheduler
        self.callbacks = callbacks

        self.logs = defaultdict(list)
        self.device = next(model.parameters()).device
        self.cancel_train = False

        os.makedirs(save_dir, exist_ok=True)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Module:
        raise NotImplementedError

    def train(self, loader: DataLoader, epoch: int) -> None:
        raise NotImplementedError

    def eval(self, loader: DataLoader) -> None:
        raise NotImplementedError

    ###-------------------------------------------------------------------------
    ### Internals
    ###-------------------------------------------------------------------------

    def _do_callbacks(self, milestone: str, args: Any) -> None:
        for callback in self.callbacks:
            milestone_fn = getattr(callback, milestone)
            milestone_fn(self, args)
