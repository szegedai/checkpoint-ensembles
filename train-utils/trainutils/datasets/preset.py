###=============================================================================
### Imports
###=============================================================================

from os import path
from typing import Callable, Tuple, Optional

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from .clothing100k import Clothing100k
from .food_101n import Food101N
from .tiny_imagenet import TinyImageNet

###=============================================================================
### Implementation
###=============================================================================

def load_dataset(
    root: str,
    transform_train: Optional[Callable] = None,
    transform_test: Optional[Callable] = None
) -> Tuple[Dataset, Dataset]:
    dataset_name = root.split('/')[-1]
    match dataset_name:
        case 'cifar10':
            return load_cifar10(root, transform_train, transform_test)
        case 'cifar100':
            return load_cifar100(root, transform_train, transform_test)
        case 'imagenet':
            return load_imagenet(root, transform_train, transform_test)
        case 'tinyimagenet':
            return load_tinyimagenet(root, transform_train, transform_test)
        case 'food-101n':
            return load_food101n(root, transform_train, transform_test)
        case 'clothing100k':
            return load_clothing100k(root, transform_train, transform_test)
        case _:
            raise ValueError(f'Unknown dataset: {dataset_name}')


def load_cifar10(
    root: str,
    transform_train: Optional[Callable] = None,
    transform_test: Optional[Callable] = None
) -> Tuple[Dataset, Dataset]:
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

    train_set = CIFAR10(root=root,
                        train=True,
                        transform=transform_train,
                        download=True)

    test_set = CIFAR10(root=root,
                       train=False,
                       transform=transform_test,
                       download=True)

    return train_set, test_set


def load_cifar100(
    root: str,
    transform_train: Optional[Callable] = None,
    transform_test: Optional[Callable] = None
) -> Tuple[Dataset, Dataset]:
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

    train_set = CIFAR100(root=root,
                         train=True,
                         transform=transform_train,
                         download=True)

    test_set = CIFAR100(root=root,
                        train=False,
                        transform=transform_test,
                        download=True)

    return train_set, test_set


def load_imagenet(
    root: str,
    transform_train: Optional[Callable] = None,
    transform_test: Optional[Callable] = None
) -> Tuple[Dataset, Dataset]:
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    train_set = ImageFolder(root=path.join(root, 'training_data'),
                            transform=transform_train)

    test_set = ImageFolder(root=path.join(root, 'validation_data'),
                           transform=transform_test)

    return train_set, test_set


def load_tinyimagenet(
    root: str,
    transform_train: Optional[Callable] = None,
    transform_test: Optional[Callable] = None
) -> Tuple[Dataset, Dataset]:
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

    train_set = TinyImageNet(root=root,
                             split='train',
                             transform=transform_train,
                             download=True)

    test_set = TinyImageNet(root=root,
                            split='val',
                            transform=transform_test,
                            download=True)

    return train_set, test_set


def load_food101n(
    root: str,
    transform_train: Optional[Callable] = None,
    transform_test: Optional[Callable] = None
) -> Tuple[Dataset, Dataset]:
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    train_set = Food101N(root=root,
                         split='train',
                         transform=transform_train)

    test_set = Food101N(root=root,
                        split='val',
                        transform=transform_test)

    return train_set, test_set


def load_clothing100k(
    root: str,
    transform_train: Optional[Callable] = None,
    transform_test: Optional[Callable] = None
) -> Tuple[Dataset, Dataset]:
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    train_set = Clothing100k(root=root,
                             split='train',
                             transform=transform_train)

    test_set = Clothing100k(root=root,
                            split='val',
                            transform=transform_test)

    return train_set, test_set
