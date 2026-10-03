# Based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

###=============================================================================
### Imports
###=============================================================================

from collections import OrderedDict
from typing import List

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from trainutils.models import BaseModel

###=============================================================================
### Implementation
###=============================================================================

class VGG(BaseModel):
    def __init__(
        self,
        config: str,
        batch_norm: bool = True,
        n_classes: int = 10
    ) -> None:
        super().__init__()
        self.features = self._make_layers(config, batch_norm)
        self.classifier = self._make_classifier(n_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, config: List, batch_norm: bool) -> Module:
        layers = OrderedDict()
        in_channels = 3

        n_block = 1
        n_layer = 1
        for x in config:
            if x == 'M':
                maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
                layers[f'maxpool{n_block}'] = maxpool
                n_block += 1
                n_layer = 1
            else:
                conv = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                layers[f'conv{n_block}_{n_layer}'] = conv

                if batch_norm:
                    bn = nn.BatchNorm2d(x)
                    layers[f'bn{n_block}_{n_layer}'] = bn

                relu = nn.ReLU()
                layers[f'relu{n_block}_{n_layer}'] = relu

                in_channels = x
                n_layer += 1

        return nn.Sequential(layers)

    def _make_classifier(self, n_classes: int) -> Module:
        classifier = OrderedDict()

        classifier['fc1'] = nn.Linear(512, 4096)
        classifier['relu1'] = nn.ReLU()
        classifier['fc2'] = nn.Linear(4096, 4096)
        classifier['relu2'] = nn.ReLU()
        classifier['fc3'] = nn.Linear(4096, n_classes)

        return nn.Sequential(classifier)

###=============================================================================
### Configs
###=============================================================================

CONFIGS = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

###=============================================================================
### Networks
###=============================================================================

def vgg11(n_classes: int = 10) -> Module:
    return VGG(CONFIGS['VGG11'], batch_norm=False, n_classes=n_classes)


def vgg11_bn(n_classes: int = 10) -> Module:
    return VGG(CONFIGS['VGG11'], batch_norm=True, n_classes=n_classes)


def vgg13(n_classes: int = 10) -> Module:
    return VGG(CONFIGS['VGG13'], batch_norm=False, n_classes=n_classes)


def vgg13_bn(n_classes: int = 10) -> Module:
    return VGG(CONFIGS['VGG13'], batch_norm=True, n_classes=n_classes)


def vgg16(n_classes: int = 10) -> Module:
    return VGG(CONFIGS['VGG16'], batch_norm=False, n_classes=n_classes)


def vgg16_bn(n_classes: int = 10) -> Module:
    return VGG(CONFIGS['VGG16'], batch_norm=True, n_classes=n_classes)


def vgg19(n_classes: int = 10) -> Module:
    return VGG(CONFIGS['VGG19'], batch_norm=False, n_classes=n_classes)


def vgg19_bn(n_classes: int = 10) -> Module:
    return VGG(CONFIGS['VGG19'], batch_norm=True, n_classes=n_classes)
