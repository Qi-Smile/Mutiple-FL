from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

# Suppress Pillow deprecation warning from torchvision datasets
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*mode.*deprecated.*Pillow.*")


@dataclass
class DatasetBundle:
    train: Dataset
    test: Dataset


def get_vision_transforms(name: str) -> Tuple[Callable, Callable]:
    """Return default train/test transforms for supported datasets."""
    name = name.lower()
    if name in {"mnist", "fashionmnist"}:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        return transform, transform
    if name in {"cifar10", "cifar-10"}:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        return train_transform, test_transform
    raise ValueError(f"Transforms for dataset '{name}' are not predefined.")


def load_torchvision_dataset(name: str, root: str = "./data", download: bool = True) -> DatasetBundle:
    """Load a torchvision dataset with sensible defaults."""
    name = name.lower()
    train_transform, test_transform = get_vision_transforms(name)

    if name == "mnist":
        train = datasets.MNIST(root=root, train=True, download=download, transform=train_transform)
        test = datasets.MNIST(root=root, train=False, download=download, transform=test_transform)
    elif name == "fashionmnist":
        train = datasets.FashionMNIST(
            root=root, train=True, download=download, transform=train_transform
        )
        test = datasets.FashionMNIST(
            root=root, train=False, download=download, transform=test_transform
        )
    elif name in {"cifar10", "cifar-10"}:
        train = datasets.CIFAR10(root=root, train=True, download=download, transform=train_transform)
        test = datasets.CIFAR10(root=root, train=False, download=download, transform=test_transform)
    else:
        raise ValueError(f"Dataset '{name}' not supported out-of-the-box.")

    return DatasetBundle(train=train, test=test)


def subset_dataset(dataset: Dataset, indices: list[int]) -> Subset:
    """Create a subset dataset for provided indices."""
    return Subset(dataset, indices)
