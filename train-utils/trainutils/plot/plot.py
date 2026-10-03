###=============================================================================
### Imports
###=============================================================================

import math
from os import path
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from pandas import DataFrame

###=============================================================================
### Implementation
###=============================================================================

def plot_train_logs(dirpath: str, metric: Union[str, List] = 'loss') -> None:
    fpath = path.join(dirpath, '_logs_training.csv')
    logs = pd.read_csv(fpath, delimiter=';')

    if type(metric) is str:
        _, ax = plt.subplots(figsize=(10, 6))
        _plot_train_logs_metric(ax, logs, metric)
    else:
        n_rows = math.ceil(len(metric) / 2)
        _, axes = plt.subplots(n_rows, 2, figsize=(20, 6*n_rows))
        for ax, m in zip(axes, metric):
            _plot_train_logs_metric(ax, logs, m)

    plt.show()

###-----------------------------------------------------------------------------
### Internals
###-----------------------------------------------------------------------------

def _plot_train_logs_metric(ax: Axes, logs: DataFrame, metric: str) -> None:
    ax.plot(logs[f'train_{metric}'], label=f'train')
    ax.plot(logs[f'val_{metric}'], label=f'val')

    if metric.endswith('accuracy'):
        ax.set_ylim(-0.01, 1.01)

    ax.grid()
    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
