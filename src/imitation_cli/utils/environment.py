import dataclasses

import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclasses.dataclass
class Config:
    env_name: str = MISSING  # The environment to train on
    n_envs: int = 8  # number of environments in VecEnv
    parallel: bool = False  # Use SubprocVecEnv rather than DummyVecEnv  TODO: when setting this to true this is really slow for some reason
    max_episode_steps: int = MISSING  # Set to positive int to limit episode horizons
    env_make_kwargs: dict = dataclasses.field(
        default_factory=dict
    )  # The kwargs passed to `spec.make`.


def make_venv(
    environment_config: Config,
    rnd: np.random.Generator,
    log_dir=None,
    **kwargs,
):
    from imitation.util import util

    return util.make_vec_env(
        **environment_config,
        rng=rnd,
        log_dir=log_dir,
        **kwargs,
    )


def make_rollout_venv(
    environment_config: Config,
    rnd: np.random.Generator,
):
    from imitation.data import wrappers

    return make_venv(
        environment_config,
        rnd,
        log_dir=None,
        post_wrappers=[lambda env, i: wrappers.RolloutInfoWrapper(env)],
    )


def register_configs(group: str):
    cs = ConfigStore.instance()
    cs.store(group=group, name="gym_env", node=Config)
