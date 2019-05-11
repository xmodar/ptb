"""Neural networks utilities module."""

from torch import nn

__all__ = ['Flatten']


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
