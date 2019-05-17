"""Datasets definitions module."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets as torchvision_datasets
from torchvision import transforms

__all__ = [
    'add_dataset', 'get_dataset', 'get_loader', 'DATASETS_DIR', 'MEANS',
    'STDS', 'NUM_CLASSES', 'IMAGE_SHAPES'
]

DATASETS_DIR = Path.home() / '.torch/datasets'
MEANS = {
    'MNIST': (0.1307,),
    'SVHN': (0.5071, 0.4867, 0.4408),
    'CIFAR10': (0.4915, 0.4823, 0.4468),
    'CIFAR100': (0.5072, 0.4867, 0.4412),
}
STDS = {
    'MNIST': (0.3081,),
    'SVHN': (0.2675, 0.2565, 0.2761),
    'CIFAR10': (0.2470, 0.2435, 0.2616),
    'CIFAR100': (0.2673, 0.2564, 0.2762),
}
IMAGE_SHAPES = {
    'MNIST': (1, 28, 28),
    'SVHN': (3, 28, 28),
    'CIFAR10': (3, 32, 32),
    'CIFAR100': (3, 32, 32),
}
NUM_CLASSES = {
    'MNIST': 10,
    'SVHN': 10,
    'CIFAR10': 10,
    'CIFAR100': 100,
}


def add_dataset(name, function):
    """Add a new dataset to the collection of datasets."""
    torchvision_datasets.__dict__[name] = function


def get_dataset(name, train=False):
    """Get a dataset given its name."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEANS[name], STDS[name]),
    ])
    kwargs = {
        'split': 'train' if train else 'test',
    } if name == 'SVHN' else {
        'train': train
    }
    return torchvision_datasets.__dict__[name](
        DATASETS_DIR, transform=transform, download=True, **kwargs)


def get_loader(dataset, train, batch_size, using_cuda, jobs):
    """Get a data loader for the specified dataset."""
    if isinstance(dataset, str):
        dataset = get_dataset(dataset, train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=jobs if using_cuda else 0,
        pin_memory=using_cuda,
        drop_last=train,
    )
