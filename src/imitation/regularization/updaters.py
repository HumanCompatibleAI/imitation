"""Implements parameter scaling algorithms to update the parameters of a regularizer."""

from typing import Protocol, Tuple, Union

import torch as th

LossType = Union[th.Tensor, float]


class LambdaUpdater(Protocol):
    """Protocol type for functions that update the regularizer parameter.

    A callable object that takes in the current lambda and the train and val loss, and
    returns the new lambda. This has been implemented as a protocol and not an ABC
    because a user might wish to provide their own implementation without having to
    inherit from the base class, e.g. by defining a function instead of a class.

    Note: if you implement `LambdaUpdater`, your implementation MUST be purely
    functional, i.e. side-effect free. The class structure should only be used
    to store constant hyperparameters. (Alternatively, closures can be used for that).
    """

    def __call__(self, lambda_, train_loss: LossType, val_loss: LossType) -> float:
        ...


class IntervalParamScaler(LambdaUpdater):
    """Scales the lambda of the regularizer by some constant factor.

    Lambda is scaled up if the ratio of the validation loss to the training loss
    is above the tolerable interval, and scaled down if the ratio is below the
    tolerable interval. Nothing happens if the ratio is within the tolerable
    interval.
    """

    def __init__(self, scaling_factor: float, tolerable_interval: Tuple[float, float]):
        """Initialize the interval parameter scaler.

        Args:
            scaling_factor: The factor by which to scale the lambda.
            tolerable_interval: The interval within which the ratio of the validation
                loss to the training loss is considered acceptable.
        """
        self.scaling_factor = scaling_factor
        self.tolerable_interval = tolerable_interval

    def __call__(
        self,
        lambda_: float,
        train_loss: LossType,
        val_loss: LossType,
    ) -> float:
        """Scales the lambda of the regularizer by some constant factor.

        Lambda is scaled up if the ratio of the validation loss to the training loss
        is above the tolerable interval, and scaled down if the ratio is below the
        tolerable interval. Nothing happens if the ratio is within the tolerable
        interval.

        Args:
            lambda_: The current value of the lambda.
            train_loss: The loss on the training set.
            val_loss: The loss on the validation set.

        Returns:
            The new value of the lambda.
        """
        # assert that the tensors val_loss and train_loss are both scalars
        assert isinstance(val_loss, float) or val_loss.dim() == 0
        assert isinstance(train_loss, float) or train_loss.dim() == 0
        val_to_train_ratio = val_loss / train_loss
        if val_to_train_ratio > self.tolerable_interval[1]:
            lambda_ *= 1 + self.scaling_factor
        elif val_to_train_ratio < self.tolerable_interval[0]:
            lambda_ *= 1 - self.scaling_factor
        return lambda_


class ConstantParamScaler(LambdaUpdater):
    """A dummy param scaler implementation to use as default."""

    def __call__(
        self,
        lambda_: float,
        train_loss: Union[float, th.Tensor],
        val_loss: Union[float, th.Tensor],
    ) -> float:
        del train_loss, val_loss
        return lambda_
