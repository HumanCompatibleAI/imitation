"""Helper methods to build and run neural networks."""
import collections
import contextlib
import functools
from abc import ABC, abstractclassmethod
from typing import Iterable, Optional, Type

import torch as th
from torch import nn


@contextlib.contextmanager
def training_mode(m: nn.Module, mode: bool = False):
    """Temporarily switch module ``m`` to specified training ``mode``.

    Args:
        m: The module to switch the mode of.
        mode: whether to set training mode (``True``) or evaluation (``False``).

    Yields:
        The module `m`.
    """
    # Modified from Christoph Heindl's method posted on:
    # https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
    old_mode = m.training
    m.train(mode)
    try:
        yield m
    finally:
        m.train(old_mode)


training = functools.partial(training_mode, mode=True)
evaluating = functools.partial(training_mode, mode=False)


class SqueezeLayer(nn.Module):
    """Torch module that squeezes a B*1 tensor down into a size-B vector."""

    def forward(self, x):
        assert x.ndim == 2 and x.shape[1] == 1
        new_value = x.squeeze(1)
        assert new_value.ndim == 1
        return new_value


class BaseNorm(nn.Module, ABC):
    """Base class for layers that try to normalize the input to mean 0 and variance 1.

    Similar to BatchNorm, LayerNorm, etc. but whereas they only use statistics from
    the current batch at train time, we use statistics from all batches.
    """

    running_mean: th.Tensor
    running_var: th.Tensor
    count: th.Tensor

    def __init__(self, num_features: int, eps: float = 1e-5):
        """Builds RunningNorm.

        Args:
            num_features: Number of features; the length of the non-batch dimension.
            eps: Small constant for numerical stability. Inputs are rescaled by
                `1 / sqrt(estimated_variance + eps)`.
        """
        super().__init__()
        self.eps = eps
        self.register_buffer("running_mean", th.empty(num_features))
        self.register_buffer("running_var", th.empty(num_features))
        self.register_buffer("count", th.empty((), dtype=th.int))
        self.reset_running_stats()

    def reset_running_stats(self) -> None:
        """Resets running stats to defaults, yielding the identity transformation."""
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.count.zero_()

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Updates statistics if in training mode. Returns normalized `x`."""
        if self.training:
            # Do not backpropagate through updating running mean and variance.
            # These updates are in-place and not differentiable. The gradient
            # is not needed as the running mean and variance are updated
            # directly by this function, and not by gradient descent.
            with th.no_grad():
                self.update_stats(x)

        return (x - self.running_mean) / th.sqrt(self.running_var + self.eps)

    @abstractclassmethod
    def update_stats(self, batch: th.Tensor) -> None:
        """Update `self.running_mean`, `self.running_var` and `self.count`."""


class RunningNorm(BaseNorm):
    """Normalizes input to mean 0 and standard deviation 1 using a running average.

    Similar to BatchNorm, LayerNorm, etc. but whereas they only use statistics from
    the current batch at train time, we use statistics from all batches.

    This should closely replicate the common practice in RL of normalizing environment
    observations, such as using `VecNormalize` in Stable Baselines.
    """

    def update_stats(self, batch: th.Tensor) -> None:
        """Update `self.running_mean`, `self.running_var` and `self.count`.

        Uses Chan et al (1979), "Updating Formulae and a Pairwise Algorithm for
        Computing Sample Variances." to update the running moments in a numerically
        stable fashion.

        Args:
            batch: A batch of data to use to update the running mean and variance.
        """
        batch_mean = th.mean(batch, dim=0)
        batch_var = th.var(batch, dim=0, unbiased=False)
        batch_count = batch.shape[0]

        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count
        self.running_mean += delta * batch_count / tot_count

        self.running_var *= self.count
        self.running_var += batch_var * batch_count
        self.running_var += th.square(delta) * self.count * batch_count / tot_count
        self.running_var /= tot_count

        self.count += batch_count


class EMANorm(BaseNorm):
    """Similar to RunningNorm but uses an exponential weighting."""

    def __init__(
        self,
        num_features: int,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        """Builds EMARunningNorm.

        Args:
            num_features: Number of features; the length of the non-batch dim.
            decay: how quickly the weight on past samples decays over time
            eps: small constant for for numerical stability.

        Raises:
            ValueError: if decay is out of range.
        """
        super().__init__(num_features, eps=eps)

        if not 0 < decay < 1:
            raise ValueError("decay must be between 0 and 1")

        self.decay = decay

    def update_stats(self, batch: th.Tensor) -> None:
        """Update `self.running_mean` and `self.running_var`.

        Reference Finch (2009), "Incremental calculation of weighted mean and variance".
            (https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf)

        Args:
            batch: A batch of data to use to update the running mean and variance.
        """
        b_size = batch.shape[0]

        if self.count == 0:
            self.running_mean = th.mean(batch, dim=0)
            if b_size > 1:
                self.running_var = th.var(batch, dim=0)
        else:
            # Shuffle the batch since we don't don't want to bias the mean
            # towards data that appears latter in the batch
            perm = th.randperm(b_size)

            alpha = 1 - self.decay

            for i in range(b_size):
                diff = batch[perm[i], ...] - self.running_mean
                incr = alpha * diff
                self.running_mean += incr
                self.running_var = self.decay * (self.running_var + diff * incr)

        self.count += b_size


def build_mlp(
    in_size: int,
    hid_sizes: Iterable[int],
    out_size: int = 1,
    name: Optional[str] = None,
    activation: Type[nn.Module] = nn.ReLU,
    dropout_prob: float = 0.0,
    squeeze_output: bool = False,
    flatten_input: bool = False,
    normalize_input_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    """Constructs a Torch MLP.

    Args:
        in_size: size of individual input vectors; input to the MLP will be of
            shape (batch_size, in_size).
        hid_sizes: sizes of hidden layers. If this is an empty iterable, then we build
            a linear function approximator.
        out_size: required size of output vector.
        name: Name to use as a prefix for the layers ID.
        activation: activation to apply after hidden layers.
        dropout_prob: Dropout probability to use after each hidden layer. If 0,
            no dropout layers are added to the network.
        squeeze_output: if out_size=1, then squeeze_input=True ensures that MLP
            output is of size (B,) instead of (B,1).
        flatten_input: should input be flattened along axes 1, 2, 3, …? Useful
            if you want to, e.g., process small images inputs with an MLP.
        normalize_input_layer: if specified, module to use to normalize inputs;
            e.g. `nn.BatchNorm` or `RunningNorm`.

    Returns:
        nn.Module: an MLP mapping from inputs of size (batch_size, in_size) to
            (batch_size, out_size), unless out_size=1 and squeeze_output=True,
            in which case the output is of size (batch_size, ).

    Raises:
        ValueError: if squeeze_output was supplied with out_size!=1.
    """
    layers = collections.OrderedDict()

    if name is None:
        prefix = ""
    else:
        prefix = f"{name}_"

    if flatten_input:
        layers[f"{prefix}flatten"] = nn.Flatten()

    # Normalize input layer
    if normalize_input_layer:
        layers[f"{prefix}normalize_input"] = normalize_input_layer(in_size)

    # Hidden layers
    prev_size = in_size
    for i, size in enumerate(hid_sizes):
        layers[f"{prefix}dense{i}"] = nn.Linear(prev_size, size)
        prev_size = size
        if activation:
            layers[f"{prefix}act{i}"] = activation()
        if dropout_prob > 0.0:
            layers[f"{prefix}dropout{i}"] = nn.Dropout(dropout_prob)

    # Final dense layer
    layers[f"{prefix}dense_final"] = nn.Linear(prev_size, out_size)

    if squeeze_output:
        if out_size != 1:
            raise ValueError("squeeze_output is only applicable when out_size=1")
        layers[f"{prefix}squeeze"] = SqueezeLayer()

    model = nn.Sequential(layers)

    return model
