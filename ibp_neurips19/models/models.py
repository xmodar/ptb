"""Neural networks definitions module."""

from torch import nn
from torchvision import models as torchvision_models

from .utils import Flatten

__all__ = ['add_model', 'get_model', 'small_cnn']


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


add_model('small_cnn', small_cnn)
