import dataclasses

from hydra.core.config_store import ConfigStore
from hydra.utils import call
from omegaconf import MISSING

from imitation_cli.utils import randomness


@dataclasses.dataclass
class Config:
    _target_: str = "imitation_cli.utils.environment.Config.make"
    env_name: str = MISSING  # The environment to train on
    n_envs: int = 8  # number of environments in VecEnv
    parallel: bool = False  # Use SubprocVecEnv rather than DummyVecEnv  TODO: when setting this to true this is really slow for some reason
    max_episode_steps: int = MISSING  # Set to positive int to limit episode horizons
    env_make_kwargs: dict = dataclasses.field(
        default_factory=dict
    )  # The kwargs passed to `spec.make`.
    rng: randomness.Config = randomness.Config

    @staticmethod
    def make(log_dir=None, **kwargs):
        from imitation.util import util

        return util.make_vec_env(log_dir=log_dir, **kwargs)


def make_rollout_venv(environment_config: Config):
    from imitation.data import wrappers

    return call(
        environment_config,
        log_dir=None,
        post_wrappers=[lambda env, i: wrappers.RolloutInfoWrapper(env)]
    )


def register_configs(group: str):
    cs = ConfigStore.instance()
    cs.store(group=group, name="gym_env", node=Config)
