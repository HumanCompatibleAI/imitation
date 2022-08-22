"""Implements a variety of regularization techniques for NN weights."""

import abc
from typing import Protocol, Tuple, Union

import torch as th
from torch import optim

from imitation.util import logger as imit_logger


class UpdateParamFn(Protocol):
    """Protocol type for functions that update the regularizer parameter.

    A callable object that takes in the current lambda and the train and val loss, and
    returns the new lambda. This has been implemented as a protocol and not an ABC
    because a user might wish to provide their own implementation without having to
    inherit from the base class, e.g. by defining a function instead of a class.

    Note: the way this is currently designed to work, calls to objects implementing
    this protocol should be fully functional, i.e. stateless and idempotent. The class
    structure should only be used to store constant hyperparameters.
    (Alternatively, closures can be used).
    """

    def __call__(self, lambda_, train_loss: th.Tensor, val_loss: th.Tensor) -> float:
        ...


class Regularizer(abc.ABC):
    """Abstract class for creating regularizers with a common interface."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        initial_lambda: float,
        update_params_fn: UpdateParamFn,
        logger: imit_logger.HierarchicalLogger,
    ) -> None:
        """Initialize the regularizer.

        Args:
            optimizer: The optimizer to which the regularizer is attached.
            initial_lambda: The initial value of the regularization parameter.
            update_params_fn: A callable object that takes in the current lambda and
                the train and val loss, and returns the new lambda.
            logger: The logger to which the regularizer will log its parameters.
        """
        self.optimizer: optim.Optimizer = optimizer
        self.lambda_: float = initial_lambda
        self.update_params_fn: UpdateParamFn = update_params_fn
        self.logger: imit_logger.HierarchicalLogger = logger

        self.logger.record("regularization_lambda", self.lambda_)

    @abc.abstractmethod
    def regularize(self, loss: th.Tensor) -> None:
        """Abstract method for performing the regularization step.

        Args:
            loss: The loss to regularize.
        """

    def update_params(self, train_loss, val_loss) -> None:
        """Update the regularization parameter.

        This method calls the update_params_fn to update the regularization parameter,
        and assigns the new value to self.lambda_. Then logs the new value using
        the provided logger.

        Args:
            train_loss: The loss on the training set.
            val_loss: The loss on the validation set.
        """
        # This avoids logging the lambda every time if we are using a constnat value.
        # It also makes the code faster as it avoids an extra function call and variable
        # assignment, even though this is probably trivial and has not been benchmarked.
        if not isinstance(self.update_params_fn, ConstantParamScaler):
            self.lambda_ = self.update_params_fn(self.lambda_, train_loss, val_loss)
            self.logger.record("regularization_lambda", self.lambda_)


class LossRegularizer(Regularizer):
    """Abstract base class for regularizers that add a loss term to the loss function.

    Requires the user to implement the _regularize_loss method.
    """

    @abc.abstractmethod
    def _regularize_loss(self, loss: th.Tensor) -> th.Tensor:
        """Implement this method to add a loss term to the loss function.

        This method should return the term to be added to the loss function,
        not the regularized loss itself.

        Args:
            loss: The loss function to which the regularization term is added.
        """
        ...

    def regularize(self, loss: th.Tensor) -> None:
        """Add the regularization term to the loss and compute gradients."""
        regularized_loss = th.add(loss, self._regularize_loss(loss))
        regularized_loss.backward()


class WeightRegularizer(Regularizer):
    """Abstract base class for regularizers that regularize the weights of a network.

    Requires the user to implement the _regularize_weight method.
    """

    @abc.abstractmethod
    def _regularize_weight(self, weight, group) -> Union[float, th.Tensor]:
        """Implement this method to regularize the weights of the network.

        This method should return the regularization term to be added to the weight,
        not the regularized weight itself.

        Args:
            weight: The weight (network parameter) to regularize.
            group: The group of parameters to which the weight belongs.
        """
        pass

    def regularize(self, loss: th.Tensor) -> None:
        """Regularize the weights of the network."""
        loss.backward()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                param.data = th.add(param.data, self._regularize_weight(param, group))


class L2Regularizer(LossRegularizer):
    """Applies L2 regularization to a loss function."""

    def _regularize_loss(self, loss: th.Tensor) -> Union[float, th.Tensor]:
        """Returns the loss penalty.

        Calculates the squared L2 norm of the weights in the optimizer,
        and returns a scaled version of it as the penalty.

        Args:
            loss: The loss to regularize.

        Returns:
            The scaled L2 norm of the weights.
        """
        penalty = 0
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                penalty += th.sum(param.data.pow(2))
        return self.lambda_ * penalty


class L1Regularizer(LossRegularizer):
    """Applies L1 regularization to a loss function."""

    def _regularize_loss(self, loss: th.Tensor) -> Union[float, th.Tensor]:
        """Returns the loss penalty.

        Calculates the L1 norm of the weights in the optimizer,
        and returns a scaled version of it as the penalty.

        Args:
            loss: The loss to regularize.

        Returns:
            The scaled L1 norm of the weights.
        """
        penalty = 0
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                penalty += th.sum(th.abs(param.data))
        return self.lambda_ * penalty


class WeightDecayRegularizer(WeightRegularizer):
    """Applies weight decay to a loss function."""

    def _regularize_weight(self, weight, group) -> Union[float, th.Tensor]:
        """Returns the weight penalty.

        Args:
            weight: The weight to regularize.
            group: The group of parameters to which the weight belongs.

        Returns:
            The weight penalty (to add to the current value of the weight)
        """
        return -self.lambda_ * group["lr"] * weight.data


class IntervalParamScaler:
    """Scales the lambda of the regularizer by some constant factor.

    Lambda is scaled up if the ratio of the validation loss to the training lossq
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
        train_loss: th.Tensor,
        val_loss: th.Tensor,
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
        # TODO(juan) does this division below work? they are tensors.
        #  Hopefully they are always of rank zero. Should we check this?
        val_to_train_ratio = val_loss / train_loss
        if val_to_train_ratio > self.tolerable_interval[1]:
            lambda_ *= 1 + self.scaling_factor
        elif val_to_train_ratio < self.tolerable_interval[0]:
            lambda_ *= 1 - self.scaling_factor
        return lambda_


class ConstantParamScaler:
    """A dummy param scaler implementation to use as default."""

    def __call__(
        self,
        lambda_: float,
        train_loss: th.Tensor,
        val_loss: th.Tensor,
    ) -> float:
        return lambda_
