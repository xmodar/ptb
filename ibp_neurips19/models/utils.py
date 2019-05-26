"""Neural networks utilities module."""

from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'Flatten', 'adjust_sequential_cnn', 'compute_output_bounds',
    'deep_mind_bounds', 'propagate_bounds'
]


def check_type(layer, types):
    """Check if a given layer is a monotonic function."""
    for t in types:
        if type(t) is type:  # pylint: disable=unidiomatic-typecheck
            if isinstance(layer, t):
                return True
        elif layer is t:
            return True
    return False


check_relu = lambda layer: check_type(layer, [F.relu, nn.ReLU])
check_monotonic = lambda layer: check_type(layer, [
    F.tanh, F.sigmoid, F.relu, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d
])


class Flatten(nn.Module):
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


def adjust_sequential_cnn(model, channels, height, width, num_classes):
    """Fix the number of hidden units in a sequential CNN."""
    if not isinstance(model, nn.Sequential):
        return model
    example_image = torch.randn(1, channels, height, width)
    if model[0].in_channels != channels:
        model[0] = nn.Conv2d(
            channels,
            model[0].out_channels,
            model[0].kernel_size,
            model[0].stride,
            model[0].padding,
            model[0].dilation,
            model[0].groups,
            model[0].bias is not None,
            model[0].padding_mode,
        )
    for i, layer in enumerate(model):
        if isinstance(layer, Flatten):
            if len(model) > i + 1 and isinstance(model[i + 1], nn.Linear):
                units = model[:i + 1](example_image).size(1)
                if model[i + 1].in_features != units:
                    model[i + 1] = nn.Linear(units, model[i + 1].out_features,
                                             model[i + 1].bias is not None)
            break
    if isinstance(model[-1], nn.Linear):
        if model[-1].out_features != num_classes:
            model[-1] = nn.Linear(model[-1].in_features, num_classes,
                                  model[-1].bias is not None)
    return model.train(model.training)


def compute_output_bounds(layer, lower, upper):
    """Compute the output bounds of a given layer."""
    if isinstance(layer, nn.Linear):
        A = layer.weight
        m = layer((upper + lower) / 2)
        r = F.linear((upper - lower) / 2, A.abs())
        return m - r, m + r
    if isinstance(layer, nn.Conv2d):
        A = layer.weight
        m = layer((upper + lower) / 2)
        r = F.conv2d(
            (upper - lower) / 2,
            A.abs(),
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )
        return m - r, m + r
    if isinstance(layer, Flatten) or check_monotonic(layer):
        return layer(lower), layer(upper)
    raise NotImplementedError(f'Unknown layer type: {layer}')


def deep_mind_bounds(sequential_model, inputs, epsilon, worst_mask=False):
    """Propagate interval bounds through a series of layers with DM method."""
    outputs = inputs if worst_mask else None
    lower, upper = inputs - epsilon, inputs + epsilon
    for layer in sequential_model:
        if worst_mask:
            if isinstance(layer, nn.Linear):
                outputs = F.linear(outputs, layer.weight, bias=layer.bias)
            elif isinstance(layer, nn.Conv2d):
                outputs = F.conv2d(
                    outputs,
                    layer.weight,
                    bias=layer.bias,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups)
            elif isinstance(layer, nn.ReLU):
                outputs[upper <= 0] = 0
            elif isinstance(layer, Flatten):
                outputs = layer(outputs)
            else:
                raise NotImplementedError(f'Unknown layer type: {layer}')
        lower, upper = compute_output_bounds(layer, lower, upper)
    return propagate_bounds.output_type(lower, upper, outputs, None)


def propagate_bounds(sequential_model, inputs, epsilon):
    """Propagate interval bounds through a series of layers."""
    ones = None
    offsets = []
    grad = torch.is_grad_enabled()
    with torch.enable_grad():
        inputs.requires_grad = True
        epsilon *= torch.ones_like(inputs)
        bounds = deep_mind_bounds(sequential_model, inputs, epsilon, True)
        for i, outputs in enumerate(bounds.midpoint.flatten(1).t(), 1):
            if ones is None:
                ones = torch.ones_like(outputs)
                end = int(bounds.midpoint.numel() / ones.numel()) + grad
            jacobian, = torch.autograd.grad(
                outputs, inputs, ones, retain_graph=i < end, create_graph=grad)
            offsets.append((jacobian.abs() * epsilon).flatten(1).sum(1))
        offsets = torch.stack(offsets, dim=1).view(bounds.midpoint.size())
    return propagate_bounds.output_type(bounds.lower, bounds.upper,
                                        bounds.midpoint, offsets)


propagate_bounds.output_type = namedtuple(
    'propagated_bounds', ['lower', 'upper', 'midpoint', 'offset'])
