###=============================================================================
### Imports
###=============================================================================

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from trainutils import utils
from trainutils.datasets import load_dataset
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

    # create save folder
    os.makedirs(config.save_dir, exist_ok=True)

    # load dataset
    transform_train = transforms.ToTensor()
    train_set, _ = load_dataset(config.data, transform_train=transform_train)

    # add noise
    if config.noise:
        train_set.targets = np.load(config.noise)

    train_loader = DataLoader(train_set,
                              batch_size=128,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True)

    # load models
    model_path = f'{config.model_dir}/models.pkl'
    models = utils.deserialize(model_path)
    models = sorted(models, key=lambda x: x['loss'])
    print('Losses:', [(m['epoch'], m['loss']) for m in models])

    # evaluate models
    results = list()
    for i in range(10):
        model = get_model(config.model, n_classes=len(train_set.classes))
        model.load_state_dict(models[i]['model'])
        model.to(device)
        model.eval()

        subresults = list()
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                output = F.softmax(model(data), dim=1)
                for o in output:
                    subresults.append(o.tolist())

        results.append(subresults)

    # top 10 mean
    has_issue = list()
    for sample_idx in range(len(train_set)):
        target = train_set.targets[sample_idx]
        outputs = [results[i][sample_idx] for i in range(10)]
        mean_softmax = np.mean(outputs, axis=0)
        vote = np.argmax(mean_softmax)
        has_issue.append(vote != target)
    clean_idxs = np.nonzero(~np.array(has_issue))[0]
    np.save(f'{config.save_dir}/filter_baseline_top10_mean.npy', clean_idxs)

    # top 10 majority voting
    has_issue = list()
    for sample_idx in range(len(train_set)):
        target = train_set.targets[sample_idx]
        outputs = [results[i][sample_idx] for i in range(10)]
        votes = [torch.argmax(torch.tensor(o)).ne(target) for o in outputs]
        vote = sum(votes) > (10 // 2)
        has_issue.append(vote)
    clean_idxs = np.nonzero(~np.array(has_issue))[0]
    np.save(f'{config.save_dir}/filter_baseline_top10_vote.npy', clean_idxs)

    # top 5 mean
    has_issue = list()
    for sample_idx in range(len(train_set)):
        target = train_set.targets[sample_idx]
        outputs = [results[i][sample_idx] for i in range(5)]
        mean_softmax = np.mean(outputs, axis=0)
        vote = np.argmax(mean_softmax)
        has_issue.append(vote != target)
    clean_idxs = np.nonzero(~np.array(has_issue))[0]
    np.save(f'{config.save_dir}/filter_baseline_top5_mean.npy', clean_idxs)

    # top 5 majority voting
    has_issue = list()
    for sample_idx in range(len(train_set)):
        target = train_set.targets[sample_idx]
        outputs = [results[i][sample_idx] for i in range(5)]
        votes = [torch.argmax(torch.tensor(o)).ne(target) for o in outputs]
        vote = sum(votes) > (5 // 2)
        has_issue.append(vote)
    clean_idxs = np.nonzero(~np.array(has_issue))[0]
    np.save(f'{config.save_dir}/filter_baseline_top5_vote.npy', clean_idxs)

    # top 1
    has_issue = list()
    for sample_idx in range(len(train_set)):
        target = train_set.targets[sample_idx]
        output = results[0][sample_idx]
        vote = torch.argmax(torch.tensor(output)).ne(target)
        has_issue.append(vote)
    clean_idxs = np.nonzero(~np.array(has_issue))[0]
    np.save(f'{config.save_dir}/filter_baseline_top1.npy', clean_idxs)

###=============================================================================
### Run
###=============================================================================

if __name__ == '__main__':
    args = sys.argv[1:]
    config = parse_args(args)
    run(config)
