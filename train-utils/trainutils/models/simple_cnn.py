# Based on https://gitlab.com/harvard-machine-learning/double-descent/-/blob/master/models/mcnn.py

###=============================================================================
### Imports
###=============================================================================

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trainutils.models import BaseModel

###=============================================================================
### Implementation
###=============================================================================

class SimpleCNN(BaseModel):
    def __init__(self, c: int = 64, n_classes: int = 10) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c*2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c*2)
        self.conv3 = nn.Conv2d(c*2, c*4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(c*4)
        self.conv4 = nn.Conv2d(c*4, c*8, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(c*8)
        self.fc1 = nn.Linear(c*8, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out
