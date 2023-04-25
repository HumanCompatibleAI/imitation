"""Configuration for Gym environments."""
from __future__ import annotations

import dataclasses
import typing
from typing import Optional, Union, cast

if typing.TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv

from hydra.core.config_store import ConfigStore
from hydra.utils import call
from omegaconf import MISSING

from imitation_cli.utils import randomness


@dataclasses.dataclass
class Config:
    """Configuration for Gym environments."""

    _target_: str = "imitation_cli.utils.environment.Config.make"
    env_name: str = MISSING  # The environment to train on
    n_envs: int = 8  # number of environments in VecEnv
    # TODO: when setting this to true this is really slow for some reason
    parallel: bool = False  # Use SubprocVecEnv rather than DummyVecEnv
    max_episode_steps: int = MISSING  # Set to positive int to limit episode horizons
    env_make_kwargs: dict = dataclasses.field(
        default_factory=dict,
    )  # The kwargs passed to `spec.make`.
    rng: randomness.Config = MISSING

    @staticmethod
    def make(log_dir: Optional[str] = None, **kwargs) -> VecEnv:
        from imitation.util import util

        return util.make_vec_env(log_dir=log_dir, **kwargs)


def make_rollout_venv(environment_config: Config) -> VecEnv:
    from imitation.data import wrappers

    return call(
        environment_config,
        log_dir=None,
        post_wrappers=[lambda env, i: wrappers.RolloutInfoWrapper(env)],
    )


def register_configs(
    group: str,
    default_rng: Union[randomness.Config, str] = MISSING,
):
    default_rng = cast(randomness.Config, default_rng)
    cs = ConfigStore.instance()
    cs.store(group=group, name="gym_env", node=Config(rng=default_rng))
    cs.store(
        group=group,
        name="cartpole",
        node=Config(env_name="CartPole-v0", max_episode_steps=500, rng=default_rng),
    )
    cs.store(
        group=group,
        name="pendulum",
        node=Config(env_name="Pendulum-v1", max_episode_steps=500, rng=default_rng),
    )
