###=============================================================================
### Imports
###=============================================================================

import os
from os import path

import torch
from torch.nn import Module

###=============================================================================
### Exports
###=============================================================================

from .base_model import BaseModel
from .lenet import LeNet
from .preact_resnet import (
    PreActResNet18,
    PreActResNet34,
    PreActResNet50,
    PreActResNet101,
    PreActResNet152
)
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .simple_cnn import SimpleCNN
from .vgg import (
    vgg11,
    vgg11_bn,
    vgg13,
    vgg13_bn,
    vgg16,
    vgg16_bn,
    vgg19,
    vgg19_bn
)

###=============================================================================
### Architectures
###=============================================================================

ARCHITECTURES = {
    'lenet': LeNet,
    'preactresnet18': PreActResNet18,
    'preactresnet34': PreActResNet34,
    'preactresnet50': PreActResNet50,
    'preactresnet101': PreActResNet101,
    'preactresnet152': PreActResNet152,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'simplecnn': SimpleCNN,
    'vgg11': vgg11,
    'vgg11_bn': vgg11_bn,
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19': vgg19,
    'vgg19_bn': vgg19_bn
}

###=============================================================================
### Implementation
###=============================================================================

def get_model(arch: str, device: torch.device = 'cpu', **kwargs) -> Module:
    model_fn = ARCHITECTURES[arch.lower()]
    model = model_fn(**kwargs)
    model.to(device)
    return model


def load_model(
    fpath: str,
    arch: str,
    device: torch.device = 'cpu',
    **kwargs
) -> Module:
    model = get_model(arch, **kwargs)
    state_dict = torch.load(fpath, map_location=device)['model']
    model.load_state_dict(state_dict)
    return model


def save_model(fpath: str, model: Module, epoch: int) -> None:
    os.makedirs(path.dirname(fpath), exist_ok=True)
    state = {'epoch': epoch, 'model': model.state_dict()}
    torch.save(state, fpath)
