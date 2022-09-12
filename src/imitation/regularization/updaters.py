"""Implements parameter scaling algorithms to update the parameters of a regularizer."""

from typing import Protocol, Tuple, Union

import numpy as np
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
            scaling_factor: The factor by which to scale the lambda, a value in (0, 1).
            tolerable_interval: The interval within which the ratio of the validation
                loss to the training loss is considered acceptable. A tuple whose first
                element is at least 0 and the second element is greater than the first.

        Raises:
            ValueError: If the tolerable interval is not a tuple of length 2.
            ValueError: if the scaling factor is not in (0, 1).
            ValueError: if the tolerable interval is negative or not a proper interval.
        """
        eps = np.finfo(float).eps
        if not (eps < scaling_factor < 1 - eps):
            raise ValueError(
                "scaling_factor must be in (0, 1) within machine precision.",
            )
        if len(tolerable_interval) != 2:
            raise ValueError("tolerable_interval must be a tuple of length 2")
        if not (0 <= tolerable_interval[0] < tolerable_interval[1]):
            raise ValueError(
                "tolerable_interval must be a tuple whose first element "
                "is at least 0 and the second element is greater than "
                "the first",
            )

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

        Raises:
            ValueError: If the loss on the validation set is not a scalar.
            ValueError: if lambda_ is zero (will result in no scaling).
            ValueError: if lambda_ is not a float.
        """
        # check that the tensors val_loss and train_loss are both scalars
        if not (
            isinstance(val_loss, float)
            or (isinstance(val_loss, th.Tensor) and val_loss.dim() == 0)
        ):
            raise ValueError("val_loss must be a scalar")

        if not (
            isinstance(train_loss, float)
            or (isinstance(train_loss, th.Tensor) and train_loss.dim() == 0)
        ):
            raise ValueError("train_loss must be a scalar")

        if np.finfo(float).eps > abs(lambda_):
            raise ValueError(
                "lambda_ must not be zero. Make sure that you're not "
                "scaling the value of lambda down too quickly or passing an "
                "initial value of zero to the lambda parameter.",
            )
        elif lambda_ < 0:
            raise ValueError("lambda_ must be non-negative")
        if not isinstance(lambda_, float):
            raise ValueError("lambda_ must be a float")
        if train_loss < 0 or val_loss < 0:
            raise ValueError("losses must be non-negative for this updater")

        eps = np.finfo(float).eps
        if train_loss < eps and val_loss < eps:
            # 0/0 is undefined, so return the current lambda
            return lambda_
        elif train_loss < eps <= val_loss:
            # the ratio would be infinite
            return lambda_ * (1 + self.scaling_factor)

        val_to_train_ratio = val_loss / train_loss
        if val_to_train_ratio > self.tolerable_interval[1]:
            lambda_ *= 1 + self.scaling_factor
        elif val_to_train_ratio < self.tolerable_interval[0]:
            lambda_ *= 1 - self.scaling_factor
        return lambda_
