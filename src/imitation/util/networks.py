"""Helper methods to build and run neural networks."""

import collections
from typing import Callable, Iterable, Optional

from torch import nn


def build_mlp(
    in_size: int,
    hid_sizes: Iterable[int],
    name: Optional[str] = None,
    activation: Optional[Callable] = nn.ReLU,
    initializer: Optional[Callable] = None,
) -> nn.Module:
    """Constructs an MLP, returning an ordered dict of layers."""
    layers = collections.OrderedDict()

    # Hidden layers
    prev_size = in_size
    for i, size in enumerate(hid_sizes):
        layers[f"{name}_dense{i}"] = nn.Linear(prev_size, size)  # type: nn.Module
        prev_size = size
        if activation:
            layers[f"{name}_act{i}"] = activation()

    # Final layer
    layers[f"{name}_dense_final"] = nn.Linear(prev_size, 1)  # type: nn.Module

    model = nn.Sequential(layers)

    return model
