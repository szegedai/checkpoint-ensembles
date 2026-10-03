###=============================================================================
### Imports
###=============================================================================

import csv
import json
import os
import pickle
from argparse import Namespace
from datetime import datetime
from os import path
from typing import Any, Dict, Optional

import torch
import random
import numpy as np

###=============================================================================
### Implementation
###=============================================================================

###-----------------------------------------------------------------------------
### Train
###-----------------------------------------------------------------------------

def get_device(gpu_id: Optional[int]) -> torch.device:
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    return device


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

###-----------------------------------------------------------------------------
### Script
###-----------------------------------------------------------------------------

def save_config(dirpath: str, config: Namespace, script_path: str) -> None:
    os.makedirs(dirpath, exist_ok=True)
    fpath = path.join(dirpath, '_config.json')
    config = {'file': script_path, 'date': datetime.now(), **vars(config)}
    with open(fpath, 'w') as f:
        json.dump(config, f, indent=4, default=str)

###-----------------------------------------------------------------------------
### Misc
###-----------------------------------------------------------------------------

def serialize(fpath: str, data: Any) -> None:
    os.makedirs(path.dirname(fpath), exist_ok=True)
    with open(fpath, 'wb') as f:
        pickle.dump(data, f)


def deserialize(fpath: str) -> Any:
    with open(fpath, 'rb') as f:
        return pickle.load(f)


def save_dict(fpath: str, dictionary: Dict) -> None:
    os.makedirs(path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(dictionary.keys())
        writer.writerows(zip(*dictionary.values()))
