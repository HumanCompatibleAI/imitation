import os

import sacred

from imitation.scripts.config.common import DEFAULT_BLANK_POLICY_KWARGS
from imitation.util import util

data_collect_ex = sacred.Experiment("data_collect")


@data_collect_ex.config
def data_collect_defaults():
    env_name = "CartPole-v1"  # The gym.Env name
    total_timesteps = int(1e6)  # Number of training timesteps in model.learn()
    num_vec = 8  # Number of environments in DummyVecEnv
    parallel = True  # Use SubprocVecEnv (generally faster if num_vec>1)
    make_blank_policy_kwargs = DEFAULT_BLANK_POLICY_KWARGS

    rollout_save = True  # Whether to save rollout files.
    rollout_save_interval = 50  # The number of training updates between saves.
    rollout_save_n_samples = 2000  # The number of timesteps saved per file.
    rollout_dir = "data/rollouts"  # The directory that rollouts are saved in.

    policy_save = False  # Whether to save policy files.
    policy_save_interval = 5  # The number of training updates between saves.
    policy_dir = "data/policies"  # The directory that policies are saved in.


@data_collect_ex.config
def logging(env_name):
    log_dir = os.path.join("output", "data_collect",
                           env_name.replace('/', '_'), util.make_timestamp())


@data_collect_ex.named_config
def ant():
    env_name = "Ant-v2"
    total_timesteps = int(2e6)


@data_collect_ex.named_config
def cartpole():
    env_name = "CartPole-v1"
    total_timesteps = int(8e5)


@data_collect_ex.named_config
def halfcheetah():
    env_name = "HalfCheetah-v2"


@data_collect_ex.named_config
def pendulum():
    env_name = "Pendulum-v0"


@data_collect_ex.named_config
def swimmer():
    env_name = "Swimmer-v2"
    make_blank_policy_kwargs = dict(
        policy_network_class=util.FeedForward64Policy,
    )


@data_collect_ex.named_config
def fast():
    """Intended for testing purposes: small # of updates, ends quickly."""
    total_timesteps = int(1e4)
