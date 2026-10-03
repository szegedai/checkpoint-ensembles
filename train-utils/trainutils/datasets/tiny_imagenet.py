###=============================================================================
### Imports
###=============================================================================

from os import path
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    download_and_extract_archive, verify_str_arg
)

###=============================================================================
### Implementation
###=============================================================================

class TinyImageNet(VisionDataset):
    base_folder = 'tiny-imagenet-200'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    zip_md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    split_list = ('train', 'val')

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform)

        self.split = verify_str_arg(split, 'split', self.split_list)

        if download:
            self._download()

        if not self._check_source():
            raise RuntimeError('Dataset not found. ' +
                               'You can use download=True to download it.')

        self._load_meta()

        if self.split == 'train':
            self.data, self.targets = self._load_train_set()
        elif self.split == 'val':
            self.data, self.targets = self._load_val_set()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path, target = self.data[index], self.targets[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return f'Split: {self.split}'

    ###-------------------------------------------------------------------------
    ### Internals
    ###-------------------------------------------------------------------------

    def _download(self) -> None:
        if self._check_source():
            print('Files already downloaded.')
            return

        download_and_extract_archive(url=self.url,
                                     download_root=self.root,
                                     filename=self.filename)

    def _check_source(self) -> bool:
        fpath = path.join(self.root, self.filename)
        dirpath = path.join(self.root, self.base_folder)
        return path.isfile(fpath) and path.isdir(dirpath)

    def _load_meta(self) -> None:
        # load wnids
        fpath = path.join(self.root, self.base_folder, 'wnids.txt')
        with open(fpath, 'r') as f:
            self.wnids = [line.strip() for line in f]
        self.wnid_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}

        # load class names
        fpath = path.join(self.root, self.base_folder, 'words.txt')
        with open(fpath, 'r') as f:
            wnid_to_words = dict(line.split('\t') for line in f)
            for wnid, words in wnid_to_words.items():
                wnid_to_words[wnid] = tuple(w.strip() for w in words.split(','))
        self.classes = [wnid_to_words[wnid] for wnid in self.wnids]
        self.class_to_idx = {wnid_to_words[wnid]: self.wnid_to_idx[wnid]
                             for wnid in self.wnids}

    def _load_train_set(self) -> Tuple[List, List]:
        data = list()
        targets = list()

        for wnid in self.wnids:
            target = self.wnid_to_idx[wnid]

            train_path = path.join(self.root, self.base_folder, 'train')
            boxes_path = path.join(train_path, wnid, f'{wnid}_boxes.txt')
            with open(boxes_path, 'r') as f:
                img_files = [line.split('\t')[0] for line in f]

            for img_file in img_files:
                img_path = path.join(train_path, wnid, 'images', img_file)
                data.append(img_path)
                targets.append(target)

        return data, targets

    def _load_val_set(self) -> Tuple[List, List]:
        data = list()
        targets = list()

        val_path = path.join(self.root, self.base_folder, 'val')
        annotation_path = path.join(val_path, 'val_annotations.txt')
        with open(annotation_path, 'r') as f:
            for line in f:
                img_file, wnid = line.split('\t')[:2]
                img_path = path.join(val_path, 'images', img_file)
                data.append(img_path)
                target = self.wnid_to_idx[wnid]
                targets.append(target)

        return data, targets
