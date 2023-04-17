import dataclasses

from hydra.core.config_store import ConfigStore
from hydra.utils import call
from omegaconf import MISSING


@dataclasses.dataclass
class Config:
    # Note: we don't define _target_ here so in the subclasses it can be defined last.
    #  This is the same pattern we use as in schedule.py.
    pass


@dataclasses.dataclass
class TanH(Config):
    _target_: str = "imitation_cli.utils.activation_function.TanH.make"

    @staticmethod
    def make():
        import torch

        return torch.nn.Tanh


@dataclasses.dataclass
class ReLU(Config):
    _target_: str = "imitation_cli.utils.activation_function.ReLU.make"

    @staticmethod
    def make():
        import torch

        return torch.nn.ReLU


@dataclasses.dataclass
class LeakyReLU(Config):
    _target_: str = "imitation_cli.utils.activation_function.LeakyReLU.make"

    @staticmethod
    def make():
        import torch

        return torch.nn.LeakyReLU


def make_activation_function(cfg: Config):
    return call(cfg)


def register_configs(group: str):
    cs = ConfigStore.instance()
    cs.store(group=group, name="tanh", node=TanH)
    cs.store(group=group, name="relu", node=ReLU)
    cs.store(group=group, name="leaky_relu", node=LeakyReLU)
