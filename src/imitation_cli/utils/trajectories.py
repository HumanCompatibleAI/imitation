import dataclasses
import pathlib

import numpy as np
from hydra.core.config_store import ConfigStore
from hydra.utils import call
from omegaconf import MISSING

from imitation_cli.utils import environment, policy


@dataclasses.dataclass
class Config:
    _target_: str = MISSING


@dataclasses.dataclass
class OnDisk(Config):
    _target_: str = "imitation_cli.utils.trajectories.OnDisk.make"
    path: pathlib.Path = MISSING

    @staticmethod
    def make(path: pathlib.Path, rng: np.random.Generator):
        from imitation.data import serialize

        serialize.load(path)


@dataclasses.dataclass
class Generated(Config):
    _target_: str = "imitation_cli.utils.trajectories.Generated.make"
    _recursive_: bool = False  # This way the expert_policy is not aut-filled.
    total_timesteps: int = int(10)  # TODO: this is low for debugging
    expert_policy: policy.Config = policy.Config(environment="${environment}")

    @staticmethod
    def make(
        total_timesteps: int, expert_policy: policy.Config, rng: np.random.Generator
    ):
        from imitation.data import rollout

        expert = policy.make_policy(expert_policy, rng)
        venv = environment.make_venv(expert_policy.environment, rng)
        return rollout.generate_trajectories(
            expert,
            venv,
            rollout.make_sample_until(min_timesteps=total_timesteps),
            rng,
            deterministic_policy=True,
        )


def get_trajectories(config: Config, rng: np.random.Generator):
    return call(config, rng=rng)


def register_configs(group: str):
    cs = ConfigStore.instance()
    cs.store(group=group, name="on_disk", node=OnDisk)
    cs.store(group=group, name="generated", node=Generated)
