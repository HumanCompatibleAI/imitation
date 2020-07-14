"""Helper methods to build and run neural networks."""

import collections
from typing import Callable, Iterable, Optional

from torch import nn


def build_mlp(
    in_size: int,
    hid_sizes: Iterable[int],
    out_size: int = 1,
    name: Optional[str] = None,
    activation: Optional[Callable] = nn.ReLU,
    initializer: Optional[Callable] = None,
) -> nn.Module:
    """Constructs an MLP, returning an ordered dict of layers."""
    layers = collections.OrderedDict()

    if initializer is not None:
        # FIXME(sam): should either remove "initializer" argument or add
        # support for custom init before committing this
        raise NotImplementedError("build_mlp doesn't yet support custom init")

    if name is None:
        prefix = ""
    else:
        prefix = f"{name}_"

    # Hidden layers
    prev_size = in_size
    for i, size in enumerate(hid_sizes):
        layers[f"{prefix}dense{i}"] = nn.Linear(prev_size, size)  # type: nn.Module
        prev_size = size
        if activation:
            layers[f"{prefix}act{i}"] = activation()

    # Final layer
    layers[f"{prefix}dense_final"] = nn.Linear(prev_size, out_size)  # type: nn.Module

    model = nn.Sequential(layers)

    return model
