"""Register optimizer classes with Hydra."""
import dataclasses

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclasses.dataclass
class Config:
    """Base config for optimizer classes."""

    _target_: str = MISSING


@dataclasses.dataclass
class Adam(Config):
    """Config for Adam optimizer class."""

    _target_: str = "imitation_cli.utils.optimizer_class.Adam.make"

    @staticmethod
    def make() -> type:
        import torch

        return torch.optim.Adam


@dataclasses.dataclass
class SGD(Config):
    """Config for SGD optimizer class."""

    _target_: str = "imitation_cli.utils.optimizer_class.SGD.make"

    @staticmethod
    def make() -> type:
        import torch

        return torch.optim.SGD


def register_configs(group: str):
    cs = ConfigStore.instance()
    cs.store(group=group, name="adam", node=Adam)
    cs.store(group=group, name="sgd", node=SGD)
