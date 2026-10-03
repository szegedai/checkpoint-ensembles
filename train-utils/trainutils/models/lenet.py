# Based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py

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

class LeNet(BaseModel):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        return out
