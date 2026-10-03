###=============================================================================
### Imports
###=============================================================================

from torch.utils.data import Dataset

###=============================================================================
### Implementation
###=============================================================================

class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (*self.dataset[index], index)
