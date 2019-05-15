"""Neural networks definitions module."""

from torch import nn
from torchvision import models as torchvision_models

from .utils import Flatten

__all__ = ['add_model', 'get_model', 'small_cnn', 'medium_cnn', 'large_cnn']


def add_model(name, function):
    """Add a new model to the collection of models."""
    torchvision_models.__dict__[name] = function


def get_model(name, pretrained=False):
    """Get a neural network given its name."""
    return torchvision_models.__dict__[name](pretrained=pretrained)


def small_cnn(pretrained=False):
    """Define a small CNN."""
    if pretrained:
        raise NotImplementedError('We don\'t have pretrained weights.')
    activation = nn.ReLU(inplace=True)
    net = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2),
        activation,
        nn.Conv2d(16, 32, 4, stride=1),
        activation,
        Flatten(),
        nn.Linear(3200, 100),
        activation,
        nn.Linear(100, 10),
    )
    return net


def medium_cnn(pretrained=False):
    """Define a medium CNN."""
    if pretrained:
        raise NotImplementedError('We don\'t have pretrained weights.')
    activation = nn.ReLU(inplace=True)
    net = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1),
        activation,
        nn.Conv2d(32, 32, 4, stride=2),
        activation,
        nn.Conv2d(32, 64, 3, stride=1),
        activation,
        nn.Conv2d(64, 64, 4, stride=2),
        activation,
        Flatten(),
        nn.Linear(1024, 512),
        activation,
        nn.Linear(512, 512),
        activation,
        nn.Linear(512, 10),
    )
    return net


def large_cnn(pretrained=False):
    """Define a large CNN."""
    if pretrained:
        raise NotImplementedError('We don\'t have pretrained weights.')
    activation = nn.ReLU(inplace=True)
    net = nn.Sequential(
        nn.Conv2d(1, 64, 3, stride=1, padding=1),
        activation,
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        activation,
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        activation,
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        activation,
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        activation,
        Flatten(),
        nn.Linear(25088, 200),
        activation,
        nn.Linear(200, 10),
    )
    return net


add_model('small_cnn', small_cnn)
add_model('medium_cnn', medium_cnn)
add_model('large_cnn', large_cnn)
