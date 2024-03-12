"""Configurations for stable_baselines3 schedules."""
import dataclasses

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclasses.dataclass
class Config:
    """Base configuration for schedules."""

    # Note: we don't define _target_ here so in the subclasses it can be defined last.
    #  This way we can instantiate a fixed schedule with `FixedSchedule(0.1)`.
    #  If we defined _target_ here, then we would have to instantiate a fixed schedule
    #  with `FixedSchedule(val=0.1)`. Otherwise we would set _target_ to 0.1.
    pass


@dataclasses.dataclass
class FixedSchedule(Config):
    """Configuration for a fixed schedule."""

    val: float = MISSING
    _target_: str = "stable_baselines3.common.utils.constant_fn"


@dataclasses.dataclass
class LinearSchedule(Config):
    """Configuration for a linear schedule."""

    start: float = MISSING
    end: float = MISSING
    end_fraction: float = MISSING
    _target_: str = "stable_baselines3.common.utils.get_linear_fn"


def register_configs(group: str):
    cs = ConfigStore.instance()
    cs.store(group=group, name="fixed", node=FixedSchedule)
    cs.store(group=group, name="linear", node=LinearSchedule)
