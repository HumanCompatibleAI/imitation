import dataclasses
import pathlib

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
    def make(path: pathlib.Path):
        from imitation.data import serialize

        serialize.load(path)


@dataclasses.dataclass
class Generated(Config):
    _target_: str = "imitation_cli.utils.trajectories.Generated.make"
    _recursive_: bool = False
    total_timesteps: int = int(10)  # TODO: this is low for debugging
    expert_policy: policy.Config = policy.Config(environment="${environment}")
    rng: randomness.Config = randomness.Config()

    @staticmethod
    def make(
        total_timesteps: int,
        expert_policy: policy.Config,
        rng: randomness.Config,
    ):
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
