"""Register optimizer classes with Hydra."""
import dataclasses
from enum import Enum
from typing import Type

import torch
from hydra.core.config_store import ConfigStore


class OptimizerClass(Enum):
    """Enum of optimizer classes."""

    Adam = torch.optim.Adam
    SGD = torch.optim.SGD


@dataclasses.dataclass
class Config:
    """Base config for optimizer classes."""

    optimizer_class: OptimizerClass
    _target_: str = "imitation_cli.utils.optimizer_class.Config.make"

    @staticmethod
    def make(optimizer_class: OptimizerClass) -> Type[torch.optim.Optimizer]:
        return optimizer_class.value


Adam = Config(OptimizerClass.Adam)
SGD = Config(OptimizerClass.SGD)


def register_configs(group: str):
    cs = ConfigStore.instance()
    for cls in OptimizerClass:
        cs.store(group=group, name=cls.name.lower(), node=Config(cls))
