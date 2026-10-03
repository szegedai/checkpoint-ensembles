###=============================================================================
### Imports
###=============================================================================

import argparse
import sys

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from trainutils import utils
from trainutils.datasets import load_dataset
from trainutils.datasets.utils import IndexedDataset
from trainutils.models import get_model
from trainutils.trainer import ClassifierTrainer, callback

from src import AggregatePredictions

###=============================================================================
### Arguments
###=============================================================================

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        type=str, default=None, required=True)
    parser.add_argument('--noise',
                        type=str, default=None)
    parser.add_argument('--test-noise',
                        type=str, default=None)
    parser.add_argument('--model',
                        type=str, default='resnet18')
    parser.add_argument('--batch-size',
                        type=int, default=128)
    parser.add_argument('--epoch',
                        type=int, default=201)
    parser.add_argument('--lr',
                        type=float, default=0.1)
    parser.add_argument('--momentum',
                        type=float, default=0.9)
    parser.add_argument('--weight-decay',
                        type=float, default=5e-4)
    parser.add_argument('--lr-steps',
                        type=int, default=[200], nargs='*')
    parser.add_argument('--agg-from',
                        type=int, default=50)
    parser.add_argument('--agg-to',
                        type=int, default=200)
    parser.add_argument('--gpu',
                        type=int, default=0)
    parser.add_argument('--seed',
                        type=int, default=19)
    parser.add_argument('--save-dir',
                        type=str, default='./model')

    return parser.parse_args(args)

###=============================================================================
### Functionality
###=============================================================================

def run(config):
    # set seed
    utils.set_seed(config.seed)

    # get device
    device = utils.get_device(config.gpu)

    # load dataset
    train_set, test_set = load_dataset(config.data)

    # load noise
    if config.noise is not None:
        train_set.targets = np.load(config.noise)
    if config.test_noise is not None:
        test_set.targets = np.load(config.test_noise)

    train_loader = DataLoader(IndexedDataset(train_set),
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=1,
                              pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=1,
                             pin_memory=True)

    # load train set without transformations
    train_set_wo_t, _ = load_dataset(config.data,
                                     transform_train=transforms.ToTensor())
    if config.noise is not None:
        train_set_wo_t.targets = np.load(config.noise)
    train_loader_wo_t = DataLoader(IndexedDataset(train_set_wo_t),
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True)

    # create model
    model = get_model(config.model, device, n_classes=len(train_set.classes))

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(),
                          lr=config.lr,
                          momentum=config.momentum,
                          weight_decay=config.weight_decay)

    scheduler = MultiStepLR(optimizer, config.lr_steps)

    callbacks = [
        callback.PrintLogs(),
        callback.SaveLogs(),
        callback.SaveCheckpoints(1000),
        AggregatePredictions(train_loader_wo_t,
                                    config.agg_from,
                                    config.agg_to)
    ]

    # train
    trainer = ClassifierTrainer(model=model,
                                criterion=criterion,
                                optimizer=optimizer,
                                n_epochs=config.epoch,
                                save_dir=config.save_dir,
                                scheduler=scheduler,
                                callbacks=callbacks)

    trainer.fit(train_loader, test_loader)

    # save config
    utils.save_config(config.save_dir, config, sys.argv[0])

###=============================================================================
### Run
###=============================================================================

if __name__ == '__main__':
    args = sys.argv[1:]
    config = parse_args(args)
    run(config)
