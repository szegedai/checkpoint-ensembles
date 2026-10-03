# based on: https://github.com/asappresearch/aum/blob/master/examples/paper_replication/runner.py#L253

###=============================================================================
### Imports
###=============================================================================

import argparse
import os
import sys

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor

from trainutils import utils
from trainutils.datasets import TinyImageNet

###=============================================================================
### Arguments
###=============================================================================

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir',
                        type=str, required=True)
    parser.add_argument('--dataset',
                        type=str, required=True)
    parser.add_argument('--split',
                        type=str, default='train')
    parser.add_argument('--noise-type',
                        type=str, default='uniform')
    parser.add_argument('--noise-rate',
                        type=float, default=0.2)
    parser.add_argument('--seed',
                        type=int, default=19)
    parser.add_argument('--save-path',
                        type=str, required=True)

    return parser.parse_args(args)

###=============================================================================
### Functionality
###=============================================================================

def run(config):
    # Set seed
    utils.set_seed(config.seed)

    # Assertions
    if config.dataset not in ['cifar10', 'cifar100', 'tinyimagenet']:
        raise ValueError('Unknown dataset')
    if config.noise_type not in ['uniform', 'flip', 'instance']:
        raise ValueError('Unknown noise type')
    if config.noise_rate <= 0.0 and config.noise_rate > 1.0:
        raise ValueError('Invalid noise rate')

    # Load dataset
    if config.dataset == 'cifar10':
        split = True if config.split == 'train' else False
        train_set = CIFAR10(config.data_dir, train=split, download=True, transform=ToTensor())
        n_classes = len(train_set.classes)
        real_targets = torch.tensor(train_set.targets)
    if config.dataset == 'cifar100':
        split = True if config.split == 'train' else False
        train_set = CIFAR100(config.data_dir, train=split, download=True, transform=ToTensor())
        n_classes = len(train_set.classes)
        real_targets = torch.tensor(train_set.targets)
    if config.dataset == 'tinyimagenet':
        train_set = TinyImageNet(config.data_dir, split=config.split, download=True, transform=ToTensor())
        n_classes = len(train_set.classes)
        real_targets = torch.tensor(train_set.targets)

    # Generate noise
    transition_mtx = torch.eye(n_classes)

    if config.noise_type == 'uniform':
        transition_mtx.mul_(1 - config.noise_rate * (n_classes / (n_classes - 1)))
        transition_mtx.add_(config.noise_rate / (n_classes - 1))
        sample_probs = transition_mtx[real_targets, :]
        noisy_targets = torch.distributions.Categorical(probs=sample_probs).sample()
    elif config.noise_type == 'flip':
        source_classes = torch.arange(n_classes)
        target_classes = (source_classes + 1).fmod(n_classes)
        transition_mtx.mul_(1 - config.noise_rate)
        transition_mtx[source_classes, target_classes] = config.noise_rate
        sample_probs = transition_mtx[real_targets, :]
        noisy_targets = torch.distributions.Categorical(probs=sample_probs).sample()
    elif config.noise_type == 'instance':
        q_dist = scipy.stats.truncnorm((0 - config.noise_rate) / 0.1,
                                       (1 - config.noise_rate) / 0.1,
                                       loc=config.noise_rate,
                                       scale=0.1)
        q = q_dist.rvs(len(train_set))

        n_features = np.prod(train_set[0][0].shape)
        w = np.random.randn(n_classes, n_features, n_classes)

        P = []
        for i, (x, y) in enumerate(zip(train_set.data, train_set.targets)):
            x = np.asanyarray(x)
            p = np.matmul(x.flatten(), w[y])
            p[y] = -np.inf
            p = q[i] * F.softmax(torch.tensor(p), dim=0).numpy()
            p[y] += 1 - q[i]
            P.append(torch.tensor(p))

        P = torch.stack(P, 0).cpu().numpy()
        noisy_targets = torch.tensor([np.random.choice(range(n_classes), p=P[i])
                                      for i in range(len(train_set))])

    print(f'Generated noise: '
          f'{(real_targets != noisy_targets).sum() / real_targets.size(dim=0)}')

    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    np.save(config.save_path, noisy_targets.numpy())

###=============================================================================
### Run
###=============================================================================

if __name__ == '__main__':
    args = sys.argv[1:]
    config = parse_args(args)
    run(config)
