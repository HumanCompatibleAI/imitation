"""Configurable trajectory sources."""
from __future__ import annotations

import dataclasses
import pathlib
import typing
from typing import Optional, Sequence, Union, cast

if typing.TYPE_CHECKING:
    from stable_baselines3.common.policies import BasePolicy
    from imitation.data.types import Trajectory
    import numpy as np

from hydra.core.config_store import ConfigStore
from hydra.utils import call
from omegaconf import MISSING

from imitation_cli.utils import environment as environment_cfg
from imitation_cli.utils import policy, randomness


@dataclasses.dataclass
class Config:
    """Base configuration for trajectory sources."""

    _target_: str = MISSING


@dataclasses.dataclass
class OnDisk(Config):
    """Configuration for loading trajectories from disk."""

    _target_: str = "imitation_cli.utils.trajectories.OnDisk.make"
    path: pathlib.Path = MISSING

    @staticmethod
    def make(path: pathlib.Path) -> Sequence[Trajectory]:
        from imitation.data import serialize

        return serialize.load(path)


@dataclasses.dataclass
class Generated(Config):
    """Configuration for generating trajectories from an expert policy."""

    _target_: str = "imitation_cli.utils.trajectories.Generated.make"
    # Note: We disable the recursive flag, so we can extract
    # the environment from the expert policy
    _recursive_: bool = False
    total_timesteps: int = MISSING
    expert_policy: policy.Config = MISSING
    rng: randomness.Config = MISSING

    @staticmethod
    def make(
        total_timesteps: int,
        expert_policy: BasePolicy,
        rng: np.random.Generator,
    ) -> Sequence[Trajectory]:
        from imitation.data import rollout

        expert = call(expert_policy)
        env = call(expert_policy.environment)
        rng = call(rng)
        return rollout.generate_trajectories(
            expert,
            env,
            rollout.make_sample_until(min_timesteps=total_timesteps),
            rng,
            deterministic_policy=True,
        )


def register_configs(
    group: str,
    default_environment: Optional[Union[environment_cfg.Config, str]] = MISSING,
    default_rng: Optional[Union[randomness.Config, str]] = MISSING,
):
    default_environment = cast(environment_cfg.Config, default_environment)
    default_rng = cast(randomness.Config, default_rng)

    cs = ConfigStore.instance()
    cs.store(group=group, name="on_disk", node=OnDisk)
    cs.store(group=group, name="generated", node=Generated(rng=default_rng))
    policy.register_configs(
        group=group + "/expert_policy",
        default_environment=default_environment,
    )
