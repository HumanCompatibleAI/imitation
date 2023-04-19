from __future__ import annotations
import dataclasses
import pathlib
import typing

if typing.TYPE_CHECKING:
    from stable_baselines3.common.policies import BasePolicy
    from imitation.data.types import Trajectory
    from typing import Sequence
    import numpy as np

from hydra.core.config_store import ConfigStore
from hydra.utils import call
from omegaconf import MISSING

from imitation_cli.utils import policy, randomness


@dataclasses.dataclass
class Config:
    _target_: str = MISSING


@dataclasses.dataclass
class OnDisk(Config):
    _target_: str = "imitation_cli.utils.trajectories.OnDisk.make"
    path: pathlib.Path = MISSING

    @staticmethod
    def make(path: pathlib.Path) -> Sequence[Trajectory]:
        from imitation.data import serialize

        return serialize.load(path)


@dataclasses.dataclass
class Generated(Config):
    _target_: str = "imitation_cli.utils.trajectories.Generated.make"
    _recursive_: bool = False  # We disable the recursive flag, so we can extract the environment from the expert policy
    total_timesteps: int = int(10)  # TODO: this is low for debugging
    expert_policy: policy.Config = MISSING
    rng: randomness.Config = randomness.Config()

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


def register_configs(group: str):
    cs = ConfigStore.instance()
    cs.store(group=group, name="on_disk", node=OnDisk)
    cs.store(group=group, name="generated", node=Generated)
