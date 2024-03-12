"""Classes for configuring activation functions."""
import dataclasses
from enum import Enum

import torch
from hydra.core.config_store import ConfigStore


class ActivationFunctionClass(Enum):
    """Enum of activation function classes."""

    TanH = torch.nn.Tanh
    ReLU = torch.nn.ReLU
    LeakyReLU = torch.nn.LeakyReLU


@dataclasses.dataclass
class Config:
    """Base class for activation function configs."""

    activation_function_class: ActivationFunctionClass
    _target_: str = "imitation_cli.utils.activation_function_class.Config.make"

    @staticmethod
    def make(activation_function_class: ActivationFunctionClass) -> type:
        return activation_function_class.value


TanH = Config(ActivationFunctionClass.TanH)
ReLU = Config(ActivationFunctionClass.ReLU)
LeakyReLU = Config(ActivationFunctionClass.LeakyReLU)


def register_configs(group: str):
    cs = ConfigStore.instance()
    for cls in ActivationFunctionClass:
        cs.store(group=group, name=cls.name.lower(), node=Config(cls))
