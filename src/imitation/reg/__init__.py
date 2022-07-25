"""Implements a variety of reguarlization techniques for NN weights."""

import abc
from typing import Dict

import torch as th
from torch import optim


class Regularizer:
    """Abstract class for creating regularizers with a common interface."""

    def __init__(self, optimizer: optim.Optimizer, params: Dict) -> None:
        """Initialize the regularizer.

        Args:
            optimizer: The optimizer to which the regularizer is attached.
            params: The initial parameters of the regularizer.
        """
        self.optimizer = optimizer
        self.params = params

    @abc.abstractmethod
    def regularize(self, loss: th.Tensor) -> None:
        pass

    @abc.abstractmethod
    def update_params(self, train_loss, val_loss):
        pass


class L2Regularizer(Regularizer):
    """Applies L2 regularization to a loss function."""

    def __init__(self, optimizer: optim.Optimizer, initial_lambda: float) -> None:
        """Initialize the regularizer.

        Args:
            optimizer: The optimizer to which the regularizer is attached.
            initial_lambda: The initial value of the regularization strength.
        """
        params = {"lambda": initial_lambda}
        super().__init__(optimizer, params)

    def regularize(self, loss: th.Tensor) -> None:
        loss_penalty = self.params["lambda"] * th.norm(loss.parameters(), p=2)
        loss = loss + loss_penalty
        loss.backward()


class WeightDecayRegularizer(Regularizer):
    """Applies weight decay to a loss function."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        initial_wd: float,
        scaling_factor: float = 0.001,
    ) -> None:
        """Initialize the regularizer.

        Args:
            optimizer: The optimizer to which the regularizer is attached.
            initial_wd: The initial value of the weight decay.
            scaling_factor: The scaling factor for the weight decay.
        """
        params = {"wd": initial_wd}
        super().__init__(optimizer, params)
        self.scaling_factor = scaling_factor

    def regularize(self, loss):
        loss.backward()
        for group in self.optimizer.param_groups():
            for param in group["params"]:
                param.data = param.data.add(
                    -self.params["wd"] * group["lr"],
                    param.data,
                )

    def update_params(self, train_loss, val_loss):
        """Dummy interpretation of Ibarz et al."""
        val_to_train_ratio = val_loss / train_loss
        if val_to_train_ratio > 1.5:
            self.params["wd"] *= 1 + self.scaling_factor
        elif val_to_train_ratio < 1.1:
            self.params["wd"] *= 1 - self.scaling_factor
