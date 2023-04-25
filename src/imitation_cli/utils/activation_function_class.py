"""Classes for configuring activation functions."""
import dataclasses

from hydra.core.config_store import ConfigStore


@dataclasses.dataclass
class Config:
    """Base class for activation function configs."""

    # Note: we don't define _target_ here so in the subclasses it can be defined last.
    #  This is the same pattern we use as in schedule.py.
    pass


@dataclasses.dataclass
class TanH(Config):
    """Config for TanH activation function."""

    _target_: str = "imitation_cli.utils.activation_function_class.TanH.make"

    @staticmethod
    def make() -> type:
        import torch

        return torch.nn.Tanh


@dataclasses.dataclass
class ReLU(Config):
    """Config for ReLU activation function."""

    _target_: str = "imitation_cli.utils.activation_function_class.ReLU.make"

    @staticmethod
    def make() -> type:
        import torch

        return torch.nn.ReLU


@dataclasses.dataclass
class LeakyReLU(Config):
    """Config for LeakyReLU activation function."""

    _target_: str = "imitation_cli.utils.activation_function_class.LeakyReLU.make"

    @staticmethod
    def make() -> type:
        import torch

        return torch.nn.LeakyReLU


def register_configs(group: str):
    cs = ConfigStore.instance()
    cs.store(group=group, name="tanh", node=TanH)
    cs.store(group=group, name="relu", node=ReLU)
    cs.store(group=group, name="leaky_relu", node=LeakyReLU)
