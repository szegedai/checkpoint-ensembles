###=============================================================================
### Imports
###=============================================================================

from os import path
from typing import Callable, Optional

from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from torchvision.datasets.utils import verify_str_arg

###=============================================================================
### Implementation
###=============================================================================

class Clothing100k(DatasetFolder):
    split_list = ('train', 'val', 'test')

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.split = verify_str_arg(split, 'split', self.split_list)

        if not self._check_source(root):
            raise RuntimeError('Dataset not found or corrupted. ' +
                               'You need to download it externally.')

        super().__init__(path.join(root, self.split),
                         default_loader,
                         IMG_EXTENSIONS,
                         transform=transform,
                         target_transform=target_transform)

        self.data = [s[0] for s in self.samples]

    def extra_repr(self) -> str:
        return f'Split: {self.split}'

    ###-------------------------------------------------------------------------
    ### Internals
    ###-------------------------------------------------------------------------

    def _check_source(self, root) -> bool:
        return path.isdir(root)
