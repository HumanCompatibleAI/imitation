import dataclasses

from hydra.core.config_store import ConfigStore
from hydra.utils import call
from omegaconf import MISSING


@dataclasses.dataclass
class Config:
    _target_: str = MISSING


@dataclasses.dataclass
class Adam(Config):
    _target_: str = "imitation_cli.utils.optimizer.Adam.make"

    @staticmethod
    def make():
        import torch

        return torch.optim.Adam


@dataclasses.dataclass
class SGD(Config):
    _target_: str = "imitation_cli.utils.optimizer.SGD.make"

    @staticmethod
    def make():
        import torch

        return torch.optim.SGD


def make_optimizer(cfg: Config):
    return call(cfg)


def register_configs(group: str):
    cs = ConfigStore.instance()
    cs.store(group=group, name="adam", node=Adam)
    cs.store(group=group, name="sgd", node=SGD)
