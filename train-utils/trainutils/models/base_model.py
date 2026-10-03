###=============================================================================
### Imports
###=============================================================================

from torch import Tensor
from torch.nn import Module

###=============================================================================
### Implementation
###=============================================================================

class BaseModel(Module):
    def forward(self, *x) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        n_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f'Trainable parameters: {n_param:,}\n\n' + super().__repr__()
