###=============================================================================
### Imports
###=============================================================================

from collections import defaultdict
from typing import Callable, Dict, Optional

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

###=============================================================================
### Implementation
###=============================================================================

def eval_model(
    model: Module,
    loader: DataLoader,
    criterion: Optional[Callable] = None
) -> Dict:
    model.eval()

    device = next(model.parameters()).device

    n_samples = 0
    results = defaultdict(lambda: 0)

    # iterate batches
    for data, target, *_ in loader:
        data, target = data.to(device), target.to(device)

        # evaluate
        with torch.no_grad():
            output = model(data)

            if criterion is not None:
                loss = criterion(output, target)
                results['loss'] += loss.item() * len(target)

            accuracy = torch.argmax(output, 1).eq(target).sum()
            results['accuracy'] += accuracy.item()

            n_samples += len(target)

    # results
    for k in results.keys():
        results[k] /= n_samples

    return dict(results)
