"""Implements a variety of reguarlization techniques for NN weights."""

import abc
from typing import Protocol, Union

import torch as th
from torch import optim

from imitation.util import logger as imit_logger


class UpdateParamFn(Protocol):
    """Protocol type for functions that update the regularizer parameter.

    A callable object that takes in the current lambda and the train and val loss, and
    returns the new lambda. This has been implemented as a protocol and not an ABC
    because a user might wish to provide their own implementation without having to
    inherit from the base class, e.g. by defining a function instead of a class. Since
    this is a perfectly valid use case and we will not be checking for isinstance(...),
    it makes no sense to restrict the type checker to ABC subclasses;
    a protocol is more appropriate.

    Note: the way this is currently designed to work, calls to objects implementing
    this protocol should be fully functional, i.e. stateless and idempotent. The class
    structure should only be used to store constant hyperparameters.
    (Alternatively, closures can be used). Unfortunately, Python does not allow typing
    "read-only" arguments (e.g. Rust lang immutables) or prevent class properties
    from being updated, so this cannot be type-checked.
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
        """
        self.optimizer: optim.Optimizer = optimizer
        self.lambda_: float = initial_lambda
        self.update_params_fn: UpdateParamFn = update_params_fn
        self.logger: imit_logger.HierarchicalLogger = logger

        self.logger.record("regularization_lambda", self.lambda_)

    @abc.abstractmethod
    def regularize(self, loss: th.Tensor) -> None:
        pass

    def update_params(self, train_loss, val_loss):
        self.lambda_ = self.update_params_fn(self.lambda_, train_loss, val_loss)
        self.logger.record("regularization_lambda", self.lambda_)


class LossRegularizer(Regularizer):
    @abc.abstractmethod
    def _regularize_loss(self, loss: th.Tensor) -> th.Tensor:
        pass

    def regularize(self, loss: th.Tensor) -> None:
        regularized_loss = th.add(loss, self._regularize_loss(loss))
        regularized_loss.backward()


class WeightRegularizer(Regularizer):
    @abc.abstractmethod
    def _regularize_weight(self, weight, group) -> Union[float, th.Tensor]:
        pass

    def regularize(self, loss: th.Tensor) -> None:
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

        Returns:
            The weight penalty (to add to the current value of the weight)
        """
        return -self.lambda_ * group["lr"] * weight.data


class IntervalParamScaler:
    def __init__(self, scaling_factor: float):
        self.scaling_factor = scaling_factor

    def __call__(
        self, lambda_: float, train_loss: th.Tensor, val_loss: th.Tensor
    ) -> float:
        # TODO(juan) does this division below work? they are tensors.
        #  Hopefully they are always of rank zero. Should we check this?
        val_to_train_ratio = val_loss / train_loss
        if val_to_train_ratio > 1.5:
            lambda_ *= 1 + self.scaling_factor
        elif val_to_train_ratio < 1.1:
            lambda_ *= 1 - self.scaling_factor
        return lambda_


class ConstantParamScaler:
    """A dummy param scaler implementation to use as default."""

    def __call__(
        self, lambda_: float, train_loss: th.Tensor, val_loss: th.Tensor
    ) -> float:
        return lambda_
