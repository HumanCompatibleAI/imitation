"""Implements the regularizer base class and some standard regularizers."""

import abc
from typing import Generic, TypeVar, Union, Optional, Type

import numpy as np
import torch as th
from torch import optim

from imitation.regularization import updaters
from imitation.util import logger as imit_logger

# this is not actually a scalar, dimension check is still required for tensor.
Scalar = Union[th.Tensor, float]

R = TypeVar("R")


class Regularizer(abc.ABC, Generic[R]):
    """Abstract class for creating regularizers with a common interface."""

    optimizer: optim.Optimizer
    lambda_: float
    lambda_updater: Optional[updaters.LambdaUpdater]
    logger: imit_logger.HierarchicalLogger

    def __init__(
        self,
        optimizer: optim.Optimizer,
        initial_lambda: float,
        lambda_updater: Optional[updaters.LambdaUpdater],
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

        if lambda_updater is None and np.allclose(initial_lambda, 0.0):
            raise ValueError(
                "If you do not pass a regularizer parameter updater your "
                "regularization strength must be non-zero, as this would "
                "result in no regularization.",
            )

        self.optimizer = optimizer
        self.lambda_ = initial_lambda
        self.lambda_updater = lambda_updater
        self.logger = logger

        self.logger.record("regularization_lambda", self.lambda_)


    @abc.abstractmethod
    def regularize(self, loss: th.Tensor) -> R:
        """Abstract method for performing the regularization step.

        The return type is a generic and the specific implementation
        must describe the meaning of the return type.

        Args:
            loss: The loss to regularize.
        """

    def update_params(self, train_loss: Scalar, val_loss: Scalar) -> None:
        """Update the regularization parameter.

        This method calls the lambda_updater to update the regularization parameter,
        and assigns the new value to `self.lambda_`. Then logs the new value using
        the provided logger.

        Args:
            train_loss: The loss on the training set.
            val_loss: The loss on the validation set.
        """
        if self.lambda_updater is not None:
            self.lambda_ = self.lambda_updater(self.lambda_, train_loss, val_loss)
            self.logger.record("regularization_lambda", self.lambda_)


class LossRegularizer(Regularizer[Scalar]):
    """Abstract base class for regularizers that add a loss term to the loss function.

    Requires the user to implement the _loss_penalty method.
    """

    @abc.abstractmethod
    def _loss_penalty(self, loss: Scalar) -> Scalar:
        """Implement this method to add a loss term to the loss function.

        This method should return the term to be added to the loss function,
        not the regularized loss itself.

        Args:
            loss: The loss function to which the regularization term is added.
        """

    def regularize(self, loss: th.Tensor) -> Scalar:
        """Add the regularization term to the loss and compute gradients.

        Args:
            loss: The loss to regularize.

        Returns:
            The regularized loss.
        """
        regularized_loss = th.add(loss, self._loss_penalty(loss))
        regularized_loss.backward()
        self.logger.record("regularized_loss", regularized_loss.item())
        return regularized_loss


class WeightRegularizer(Regularizer):
    """Abstract base class for regularizers that regularize the weights of a network.

    Requires the user to implement the _weight_penalty method.
    """

    @abc.abstractmethod
    def _weight_penalty(self, weight: th.Tensor, group: dict) -> Scalar:
        """Implement this method to regularize the weights of the network.

        This method should return the regularization term to be added to the weight,
        not the regularized weight itself.

        Args:
            weight: The weight (network parameter) to regularize.
            group: The group of parameters to which the weight belongs.
        """

    def regularize(self, loss: th.Tensor) -> None:
        """Regularize the weights of the network."""
        loss.backward()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                param.data = th.add(param.data, self._weight_penalty(param, group))


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
        if not isinstance(p, int) or p < 1:
            raise ValueError("p must be a positive integer")
        self.p = p

    def _loss_penalty(self, loss: Scalar) -> Scalar:
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
                penalty += th.linalg.vector_norm(param, ord=self.p).pow(self.p)
        return self.lambda_ * penalty


class WeightDecayRegularizer(WeightRegularizer):
    """Applies weight decay to a loss function."""

    def _weight_penalty(self, weight, group) -> Scalar:
        """Returns the weight penalty.

        Args:
            weight: The weight to regularize.
            group: The group of parameters to which the weight belongs.

        Returns:
            The weight penalty (to add to the current value of the weight)
        """
        return -self.lambda_ * group["lr"] * weight.data


class RegularizerFactory:
    """Factory class for creating regularizers."""

    @staticmethod
    def create(
        regularizer_cls: Optional[Type[Regularizer]],
        initial_lambda: float,
        lambda_updater: Optional[updaters.LambdaUpdater],
        val_split: float,
        optimizer: optim.Optimizer,
        logger: imit_logger.HierarchicalLogger,
        **kwargs,
    ) -> Optional[Regularizer]:
        """Create a regularizer.

        This method takes care of validate the regularization arguments passed to any
        network, and returns a regularizer object if a regularizer_cls is provided.

        Args:
            regularizer_cls: The class of the regularizer to create.
            initial_lambda: The initial value of the regularization parameter.
            lambda_updater: The updater for the regularization parameter.
            val_split: The fraction of the training data to use for validation.
            optimizer: The optimizer to use for training.
            logger: The logger to use for logging.
            **kwargs: Additional keyword arguments to pass to the regularizer.

        Returns:
            A regularizer object if a regularizer_cls is provided, None otherwise.
        """
        if regularizer_cls is None:
            if initial_lambda > 0:
                raise ValueError(
                    "Regularization strength is non-zero but no regularizer class "
                    "is specified.",
                )
            if lambda_updater is not None:
                raise ValueError(
                    "Regularization updater class was provided but no regularizer "
                    "class is specified.",
                )
            if val_split > 0:
                raise ValueError(
                    "Validation split is non-zero but no regularizer class "
                    "is specified.",
                )
            if kwargs:
                raise ValueError(
                    "Regularizer class is not specified but kwargs are provided.",
                )
            return None
        else:
            if lambda_updater is not None and np.allclose(val_split, 0.0):
                raise ValueError(
                    "If you pass a regularizer parameter updater, you must also "
                    "pass a non-zero value for the validation split. Otherwise "
                    "the updater won't have any validation data to use for updating.",
                )
            elif lambda_updater is None and val_split > 0:
                raise ValueError(
                    "If you pass a non-zero validation split, you must also "
                    "pass a regularizer parameter updater. Otherwise you are wasting"
                    " data into the validation split that will not be used.",
                )
            if val_split < 0 or val_split > 1:
                raise ValueError("val_split must be strictly between 0 and 1.")

            return regularizer_cls(
                initial_lambda=initial_lambda,
                optimizer=optimizer,
                lambda_updater=lambda_updater,
                logger=logger,
                **kwargs,
            )