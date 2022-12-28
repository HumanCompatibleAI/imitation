"""Helper methods to build and run neural networks."""
import abc
import collections
import contextlib
import functools
from typing import Dict, Iterable, Optional, Type, Union

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


class BaseNorm(nn.Module, abc.ABC):
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
        BaseNorm.reset_running_stats(self)

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

        # Note: this is different from the behavior in stable-baselines, see
        # https://github.com/HumanCompatibleAI/imitation/issues/442
        return (x - self.running_mean) / th.sqrt(self.running_var + self.eps)

    @abc.abstractmethod
    def update_stats(self, batch: th.Tensor) -> None:
        """Update `self.running_mean`, `self.running_var` and `self.count`."""


class RunningNorm(BaseNorm):
    """Normalizes input to mean 0 and standard deviation 1 using a running average.

    Similar to BatchNorm, LayerNorm, etc. but whereas they only use statistics from
    the current batch at train time, we use statistics from all batches.

    This should replicate the common practice in RL of normalizing environment
    observations, such as using ``VecNormalize`` in Stable Baselines. Note that
    the behavior of this class is slightly different from `VecNormalize`, e.g.,
    it works with the current reward instead of return estimate, and subtracts the mean
    reward whereas ``VecNormalize`` only rescales it.
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

    inv_learning_rate: th.Tensor
    num_batches: th.IntTensor

    def __init__(
        self,
        num_features: int,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        """Builds EMARunningNorm.

        Args:
            num_features: Number of features; the length of the non-batch dim.
            decay: how quickly the weight on past samples decays over time.
            eps: small constant for numerical stability.

        Raises:
            ValueError: if decay is out of range.
        """
        super().__init__(num_features, eps=eps)

        if not 0 < decay < 1:
            raise ValueError("decay must be between 0 and 1")

        self.decay = decay
        self.register_buffer("inv_learning_rate", th.empty(()))
        self.register_buffer("num_batches", th.empty((), dtype=th.int))
        EMANorm.reset_running_stats(self)

    def reset_running_stats(self):
        """Reset the running stats of the normalization layer."""
        super().reset_running_stats()
        self.inv_learning_rate.zero_()
        self.num_batches.zero_()

    def update_stats(self, batch: th.Tensor) -> None:
        """Update `self.running_mean` and `self.running_var` in batch mode.

        Reference Algorithm 3 from:
        https://github.com/HumanCompatibleAI/imitation/files/9456540/Incremental_batch_EMA_and_EMV.pdf

        Args:
            batch: A batch of data to use to update the running mean and variance.
        """
        b_size = batch.shape[0]
        if len(batch.shape) == 1:
            batch = batch.reshape(b_size, 1)

        self.inv_learning_rate += self.decay**self.num_batches
        learning_rate = 1 / self.inv_learning_rate

        # update running mean
        delta_mean = batch.mean(0) - self.running_mean
        self.running_mean += learning_rate * delta_mean

        # update running variance
        batch_var = batch.var(0, unbiased=False)
        delta_var = batch_var + (1 - learning_rate) * delta_mean**2 - self.running_var
        self.running_var += learning_rate * delta_var

        self.count += b_size
        self.num_batches += 1  # type: ignore[misc]


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
        out_size: size of output vector.
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
    layers: Dict[str, nn.Module] = {}

    if name is None:
        prefix = ""
    else:
        prefix = f"{name}_"

    if flatten_input:
        layers[f"{prefix}flatten"] = nn.Flatten()

    # Normalize input layer
    if normalize_input_layer:
        try:
            layer_instance = normalize_input_layer(in_size)  # type: ignore[call-arg]
        except TypeError as exc:
            raise ValueError(
                f"normalize_input_layer={normalize_input_layer} is not a valid "
                "normalization layer type accepting only one argument (in_size).",
            ) from exc
        layers[f"{prefix}normalize_input"] = layer_instance

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

    model = nn.Sequential(collections.OrderedDict(layers))

    return model


def build_cnn(
    in_channels: int,
    hid_channels: Iterable[int],
    out_size: int = 1,
    name: Optional[str] = None,
    activation: Type[nn.Module] = nn.ReLU,
    kernel_size: int = 3,
    stride: int = 1,
    padding: Union[int, str] = "same",
    dropout_prob: float = 0.0,
    squeeze_output: bool = False,
) -> nn.Module:
    """Constructs a Torch CNN.

    Args:
        in_channels: number of channels of individual inputs; input to the CNN will have
            shape (batch_size, in_size, in_height, in_width).
        hid_channels: number of channels of hidden layers. If this is an empty iterable,
            then we build a linear function approximator.
        out_size: size of output vector.
        name: Name to use as a prefix for the layers ID.
        activation: activation to apply after hidden layers.
        kernel_size: size of convolutional kernels.
        stride: stride of convolutional kernels.
        padding: padding of convolutional kernels.
        dropout_prob: Dropout probability to use after each hidden layer. If 0,
            no dropout layers are added to the network.
        squeeze_output: if out_size=1, then squeeze_input=True ensures that CNN
            output is of size (B,) instead of (B,1).

    Returns:
        nn.Module: a CNN mapping from inputs of size (batch_size, in_size, in_height,
            in_width) to (batch_size, out_size), unless out_size=1 and
            squeeze_output=True, in which case the output is of size (batch_size, ).

    Raises:
        ValueError: if squeeze_output was supplied with out_size!=1.
    """
    layers: Dict[str, nn.Module] = {}

    if name is None:
        prefix = ""
    else:
        prefix = f"{name}_"

    prev_channels = in_channels
    for i, n_channels in enumerate(hid_channels):
        layers[f"{prefix}conv{i}"] = nn.Conv2d(
            prev_channels,
            n_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        prev_channels = n_channels
        if activation:
            layers[f"{prefix}act{i}"] = activation()
        if dropout_prob > 0.0:
            layers[f"{prefix}dropout{i}"] = nn.Dropout(dropout_prob)

    # final dense layer
    layers[f"{prefix}avg_pool"] = nn.AdaptiveAvgPool2d(1)
    layers[f"{prefix}flatten"] = nn.Flatten()
    layers[f"{prefix}dense_final"] = nn.Linear(prev_channels, out_size)

    if squeeze_output:
        if out_size != 1:
            raise ValueError("squeeze_output is only applicable when out_size=1")
        layers[f"{prefix}squeeze"] = SqueezeLayer()

    model = nn.Sequential(collections.OrderedDict(layers))
    return model
