"""Utilities for the training module."""

import random

import numpy as np
import torch

__all__ = [
    'manual_seed', 'compute_accuracy', 'AverageMeter', 'Flatten',
    'get_device_order'
]


def manual_seed(value=None, benchmark_otherwise=False):
    """Seeds NumPy, PyTorch, and the builtin random number generators."""
    if value is None:
        if benchmark_otherwise:
            torch.backends.cudnn.benchmark = False
    else:
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def compute_accuracy(output, target, top_k=(1,)):
    """Compute the accuracy over the k top predictions."""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_device_order():
    """Get the cuda devices sorted from highest to lowest total memory."""
    return sorted(
        range(torch.cuda.device_count()),
        key=lambda i: -torch.cuda.get_device_properties(i).total_memory,
    )


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        """Initialize an average meter."""
        self.name = name
        self.fmt = fmt
        self.val = self.avg = self.sum = self.count = 0

    def reset(self):
        """Reset all the counters."""
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        """Update the counters."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        """Nice representation."""
        msg = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return msg.format(**self.__dict__)

    def __str__(self):
        """Short representation."""
        return f'{{{self.fmt}}}'.format(self.avg)


class Flatten(torch.nn.Module):
    """Flattens a tensor."""

    def __init__(self, start_dim=1, end_dim=-1):
        """Initialize the range of flattened dimensions."""
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(  # pylint: disable=arguments-differ
            self,
            inputs,
            start_dim=None,
            end_dim=None):
        """Flatten the input tensor."""
        if start_dim is None:
            start_dim = self.start_dim
        if end_dim is None:
            end_dim = self.end_dim
        return inputs.flatten(start_dim, end_dim)
