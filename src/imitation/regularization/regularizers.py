"""Implements the regularizer base class and some standard regularizers."""

import abc
from typing import Generic, TypeVar, Union

import torch as th
from torch import optim

from imitation.regularization import updaters
from imitation.util import logger as imit_logger

LossType = Union[th.Tensor, float]
RegularizeReturnType = TypeVar(
    "RegularizeReturnType", bound=Union[th.Tensor, float, None]
)


class Regularizer(abc.ABC, Generic[RegularizeReturnType]):
    """Abstract class for creating regularizers with a common interface."""

    optimizer: optim.Optimizer
    lambda_: float
    lambda_updater: updaters.LambdaUpdater
    logger: imit_logger.HierarchicalLogger

    def __init__(
        self,
        optimizer: optim.Optimizer,
        initial_lambda: float,
        lambda_updater: updaters.LambdaUpdater,
        logger: imit_logger.HierarchicalLogger,
    ) -> None:
        """Initialize the regularizer.

        Args:
            optimizer: The optimizer to which the regularizer is attached.
            initial_lambda: The initial value of the regularization parameter.
            lambda_updater: A callable object that takes in the current lambda and
                the train and val loss, and returns the new lambda.
            logger: The logger to which the regularizer will log its parameters.
        """
        self.optimizer = optimizer
        self.lambda_ = initial_lambda
        self.lambda_updater = lambda_updater
        self.logger = logger

        self.logger.record("regularization_lambda", self.lambda_)

    @abc.abstractmethod
    def regularize(self, loss: th.Tensor) -> RegularizeReturnType:
        """Abstract method for performing the regularization step.

        Args:
            loss: The loss to regularize.
        """
        ...

    def update_params(self, train_loss: LossType, val_loss: LossType) -> None:
        """Update the regularization parameter.

        This method calls the lambda_updater to update the regularization parameter,
        and assigns the new value to `self.lambda_`. Then logs the new value using
        the provided logger.

        Args:
            train_loss: The loss on the training set.
            val_loss: The loss on the validation set.
        """
        # This avoids logging the lambda every time if we are using a constant value.
        # It also makes the code faster as it avoids an extra function call and variable
        # assignment, even though this is probably trivial and has not been benchmarked.
        if not isinstance(self.lambda_updater, updaters.ConstantParamScaler):
            self.lambda_ = self.lambda_updater(self.lambda_, train_loss, val_loss)
            self.logger.record("regularization_lambda", self.lambda_)


class LossRegularizer(Regularizer[LossType]):
    """Abstract base class for regularizers that add a loss term to the loss function.

    Requires the user to implement the _regularize_loss method.
    """

    @abc.abstractmethod
    def _regularize_loss(self, loss: LossType) -> LossType:
        """Implement this method to add a loss term to the loss function.

        This method should return the term to be added to the loss function,
        not the regularized loss itself.

        Args:
            loss: The loss function to which the regularization term is added.
        """
        ...

    def regularize(self, loss: th.Tensor) -> LossType:
        """Add the regularization term to the loss and compute gradients."""
        regularized_loss = th.add(loss, self._regularize_loss(loss))
        regularized_loss.backward()
        self.logger.record("regularized_loss", regularized_loss.item())
        return regularized_loss


class WeightRegularizer(Regularizer[None]):
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


class LpRegularizer(LossRegularizer):
    """Applies Lp regularization to a loss function."""

    p: int

    def __init__(
        self,
        optimizer: optim.Optimizer,
        initial_lambda: float,
        lambda_updater: updaters.LambdaUpdater,
        logger: imit_logger.HierarchicalLogger,
        p: int,
    ) -> None:
        """Initialize the regularizer."""
        super().__init__(optimizer, initial_lambda, lambda_updater, logger)
        self.p = p

    def _regularize_loss(self, loss: LossType) -> LossType:
        """Returns the loss penalty.

        Calculates the p-th power of the Lp norm of the weights in the optimizer,
        and returns a scaled version of it as the penalty.

        Args:
            loss: The loss to regularize.

        Returns:
            The scaled pth power of the Lp norm of the network weights.
        """
        del loss
        penalty = 0
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                penalty += th.linalg.vector_norm(param.data, ord=self.p).pow(self.p)
        return self.lambda_ * penalty


class WeightDecayRegularizer(WeightRegularizer):
    """Applies weight decay to a loss function."""

    def _regularize_weight(self, weight, group) -> LossType:
        """Returns the weight penalty.

        Args:
            weight: The weight to regularize.
            group: The group of parameters to which the weight belongs.

        Returns:
            The weight penalty (to add to the current value of the weight)
        """
        return -self.lambda_ * group["lr"] * weight.data
