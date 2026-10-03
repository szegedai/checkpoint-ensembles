###=============================================================================
### Imports
###=============================================================================

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from trainutils import utils
from trainutils.datasets import load_dataset
from trainutils.datasets.utils import IndexedDataset
from trainutils.models import get_model

###=============================================================================
### Arguments
###=============================================================================

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        type=str, default=None, required=True)
    parser.add_argument('--model-dir',
                        type=str, default=None, required=True)
    parser.add_argument('--noise',
                        type=str, default=None)
    parser.add_argument('--model',
                        type=str, default='resnet18')
    parser.add_argument('--gpu',
                        type=int, default=0)
    parser.add_argument('--seed',
                        type=int, default=19)
    parser.add_argument('--save-dir',
                        type=str, default='./filters')

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
    transform_train = transforms.ToTensor()
    train_set, _ = load_dataset(config.data, transform_train=transform_train)

    # add noise
    if config.noise:
        train_set.targets = np.load(config.noise)

    train_loader = DataLoader(IndexedDataset(train_set),
                              batch_size=128,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True)

    # load models
    model_path = f'{config.model_dir}/models.pkl'
    models = utils.deserialize(model_path)
    models = sorted(models, key=lambda x: x['loss'])
    print('Losses:', [(m['epoch'], m['loss']) for m in models])

    # calculate gradients
    if not os.path.isfile(f'{config.save_dir}/gradients.pkl'):
        results = dict()
        criterion = nn.CrossEntropyLoss()
        for model_log in tqdm(models):
            model = get_model(config.model, n_classes=len(train_set.classes))
            model.load_state_dict(model_log['model'])
            model.to(device)
            model.eval()

            per_sample_gradients = list()
            for data, target, index in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                for y_hat, y, idx in zip(output, target, index):
                    loss = criterion(y_hat, y)
                    params = list(model.parameters())[-3]
                    gradients = torch.autograd.grad(loss, params, retain_graph=True)
                    gradients_sum = [grad.sum().item() for grad in gradients][0]
                    per_sample_gradients.append((idx.item(), gradients_sum))

            results[model_log['epoch']] = sorted(per_sample_gradients)

        utils.serialize(f'{config.save_dir}/gradients.pkl', results)
    else:
        results = utils.deserialize(f'{config.save_dir}/gradients.pkl')

    # top 10 mean
    subresults = defaultdict(list)
    for gradients in list(results.values()):
        for idx, grad in gradients:
            subresults[idx].append(grad)
    has_issue = [np.mean(gradients) > 0 for gradients in subresults.values()]
    clean_idxs = np.nonzero(~np.array(has_issue))[0]
    np.save(f'{config.save_dir}/filter_bgss_top10_mean.npy', clean_idxs)

    # top 10 majority vote
    subresults = defaultdict(list)
    for gradients in list(results.values()):
        for idx, grad in gradients:
            subresults[idx].append(grad > 0)
    has_issue = [sum(gradients) > (10//2) for gradients in subresults.values()]
    clean_idxs = np.nonzero(~np.array(has_issue))[0]
    np.save(f'{config.save_dir}/filter_bgss_top10_vote.npy', clean_idxs)

    # top 5 mean
    subresults = defaultdict(list)
    for gradients in list(results.values())[:5]:
        for idx, grad in gradients:
            subresults[idx].append(grad)
    has_issue = [np.mean(gradients) > 0 for gradients in subresults.values()]
    clean_idxs = np.nonzero(~np.array(has_issue))[0]
    np.save(f'{config.save_dir}/filter_bgss_top5_mean.npy', clean_idxs)

    # top 5 majority vote
    subresults = defaultdict(list)
    for gradients in list(results.values())[:5]:
        for idx, grad in gradients:
            subresults[idx].append(grad > 0)
    has_issue = [sum(gradients) > (5//2) for gradients in subresults.values()]
    clean_idxs = np.nonzero(~np.array(has_issue))[0]
    np.save(f'{config.save_dir}/filter_bgss_top5_vote.npy', clean_idxs)

    # top 1
    subresults = defaultdict(list)
    for gradients in list(results.values())[:1]:
        for idx, grad in gradients:
            subresults[idx].append(grad > 0)
    has_issue = [gradients for gradients in subresults.values()]
    clean_idxs = np.nonzero(~np.array(has_issue))[0]
    np.save(f'{config.save_dir}/filter_bgss_top1.npy', clean_idxs)

###=============================================================================
### Run
###=============================================================================

if __name__ == '__main__':
    args = sys.argv[1:]
    config = parse_args(args)
    run(config)
