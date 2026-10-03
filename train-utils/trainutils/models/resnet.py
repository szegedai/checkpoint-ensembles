# Based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

###=============================================================================
### Imports
###=============================================================================

from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from trainutils.models import BaseModel

###=============================================================================
### Implementation
###=============================================================================

class ResNet(BaseModel):
    def __init__(
        self,
        block: Module,
        n_blocks: int,
        n_classes: int = 10
    ) -> None:
        super().__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, n_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, n_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, n_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, n_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, n_classes)

    def _make_layer(
        self,
        block: Module,
        planes: int,
        n_blocks: int,
        stride: int
    ) -> Module:
        strides = [stride] + [1]*(n_blocks-1)
        layers = list()
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, out.shape[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class BasicBlock(Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_planes,
                                   self.expansion*planes,
                                   kernel_size=1,
                                   stride=stride,
                                   bias=False)),
                ('bn', nn.BatchNorm2d(self.expansion*planes))
            ]))

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion*planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_planes,
                                   self.expansion*planes,
                                   kernel_size=1,
                                   stride=stride,
                                   bias=False)),
                ('bn', nn.BatchNorm2d(self.expansion*planes))
            ]))

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)

        return out

###=============================================================================
### Networks
###=============================================================================

def ResNet18(n_classes: int = 10) -> Module:
    return ResNet(BasicBlock, [2, 2, 2, 2], n_classes)


def ResNet34(n_classes: int = 10) -> Module:
    return ResNet(BasicBlock, [3, 4, 6, 3], n_classes)


def ResNet50(n_classes: int = 10) -> Module:
    return ResNet(Bottleneck, [3, 4, 6, 3], n_classes)


def ResNet101(n_classes: int = 10) -> Module:
    return ResNet(Bottleneck, [3, 4, 23, 3], n_classes)


def ResNet152(n_classes: int = 10) -> Module:
    return ResNet(Bottleneck, [3, 8, 36, 3], n_classes)
