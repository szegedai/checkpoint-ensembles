###=============================================================================
### Imports
###=============================================================================

import csv
from os import path
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg

###=============================================================================
### Implementation
###=============================================================================

class Food101N(VisionDataset):
    base_folder = 'Food-101N_release'
    split_list = ('train', 'val')

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform)

        self.split = verify_str_arg(split, 'split', self.split_list)

        if not self._check_source():
            raise RuntimeError('Dataset not found or corrupted. ' +
                               'You need to download it externally.')

        self._load_meta()

        # self.data, self.targets, self.target_verification = self._load_dataset()
        self.data, self.targets = self._load_dataset()

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

    def _check_source(self) -> bool:
        dirpath = path.join(self.root, self.base_folder)
        return path.isdir(dirpath)

    def _load_meta(self) -> None:
        fpath = path.join(self.root, self.base_folder, 'meta', 'classes.txt')
        with open(fpath, 'r') as f:
            self.classes = [line.strip() for line in f][1:]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    # verified only
    # def _load_dataset(self) -> Tuple[List, List, List]:
    #     data = list()
    #     targets = list()
    #     targets_verification = list()

    #     fname = f'verified_{self.split}.tsv'
    #     fpath = path.join(self.root, self.base_folder, 'meta', fname)
    #     with open(fpath, 'r') as f:
    #         reader = csv.reader(f, delimiter='\t')
    #         _ = next(reader)

    #         images_path = path.join(self.root, self.base_folder, 'images')
    #         for img_path, is_correct in reader:
    #             img_path = path.join(images_path, img_path)

    #             class_name = path.basename(path.dirname(img_path))
    #             target = self.class_to_idx[class_name]

    #             data.append(img_path)
    #             targets.append(target)
    #             targets_verification.append(int(is_correct))

    #     return data, targets, targets_verification

    # only food-101n
    # def _load_dataset(self) -> Tuple[List, List, List]:
    #     val_fps = list()

    #     fname = 'verified_val.tsv'
    #     fpath = path.join(self.root, self.base_folder, 'meta', fname)
    #     with open(fpath, 'r') as f:
    #         reader = csv.reader(f, delimiter='\t')
    #         _ = next(reader)
    #         images_path = path.join(self.root, self.base_folder, 'images')
    #         for img_path, _ in reader:
    #             img_path = path.join(images_path, img_path)
    #             val_fps.append(img_path)

    #     data = list()
    #     targets = list()

    #     fname = 'imagelist.tsv'
    #     fpath = path.join(self.root, self.base_folder, 'meta', fname)
    #     with open(fpath, 'r') as f:
    #         reader = csv.reader(f, delimiter='\t')
    #         _ = next(reader)

    #         images_path = path.join(self.root, self.base_folder, 'images')
    #         for img_path in reader:
    #             img_path = path.join(images_path, img_path[0])

    #             if (self.split == 'val' and img_path in val_fps) or \
    #                (self.split == 'train' and img_path not in val_fps):
    #                 class_name = path.basename(path.dirname(img_path))
    #                 target = self.class_to_idx[class_name]

    #                 data.append(img_path)
    #                 targets.append(target)

    #     return data, targets

    def _load_dataset(self) -> Tuple[List, List, List]:
        if self.split == 'train':
            val_fps = list()
            fname = 'verified_val.tsv'
            fpath = path.join(self.root, self.base_folder, 'meta', fname)
            with open(fpath, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                _ = next(reader)
                images_path = path.join(self.root, self.base_folder, 'images')
                for img_path, _ in reader:
                    img_path = path.join(images_path, img_path)
                    val_fps.append(img_path)

            data = list()
            targets = list()
            fname = 'imagelist.tsv'
            fpath = path.join(self.root, self.base_folder, 'meta', fname)
            with open(fpath, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                _ = next(reader)

                images_path = path.join(self.root, self.base_folder, 'images')
                for img_path in reader:
                    img_path = path.join(images_path, img_path[0])

                    if img_path not in val_fps:
                        class_name = path.basename(path.dirname(img_path))
                        target = self.class_to_idx[class_name]

                        data.append(img_path)
                        targets.append(target)
        else:
            fname = '../food-101/meta/test.txt'
            fpath = path.join(self.root, self.base_folder, fname)
            data = list()
            targets = list()
            with open(fpath, 'r') as f:
                reader = csv.reader(f, delimiter='\t')

                images_path = path.join(self.root, 'food-101', 'images')
                for img_path in reader:
                    img_path = path.join(images_path, img_path[0] + '.jpg')

                    class_name = path.basename(path.dirname(img_path))
                    target = self.class_to_idx[class_name]

                    data.append(img_path)
                    targets.append(target)

        return data, targets
