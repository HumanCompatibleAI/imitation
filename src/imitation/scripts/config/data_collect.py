import os

import sacred

from imitation.scripts.config.common import DEFAULT_BLANK_POLICY_KWARGS
from imitation.util import util

data_collect_ex = sacred.Experiment("data_collect")


@data_collect_ex.config
def data_collect_defaults():
  env_name = "CartPole-v1"  # The gym.Env name
  total_timesteps = int(1e6)  # Number of training timesteps in model.learn()
  num_vec = 8  # Number of environments in VecEnv
  parallel = True  # Use SubprocVecEnv (generally faster if num_vec>1)
  normalize = True  # Use VecNormalize
  make_blank_policy_kwargs = dict(DEFAULT_BLANK_POLICY_KWARGS)

  rollout_save_interval = -1  # Num updates between saves (<=0 disables)
  rollout_save_final = True  # If True, save after training is finished.
  rollout_save_n_timesteps = 2000  # Min timesteps saved per file, optional.
  rollout_save_n_episodes = None  # Num episodes saved per file, optional.

  policy_save_interval = 100  # Num updates between saves (<=0 disables)
  policy_save_final = True  # If True, save after training is finished.

  log_root = os.path.join("output", "data_collect")  # output directory


@data_collect_ex.config
def logging(env_name, log_root):
  log_dir = os.path.join(log_root, env_name.replace('/', '_'),
                         util.make_timestamp())


# Standard Gym env configs

@data_collect_ex.named_config
def acrobot():
  env_name = "Acrobot-v1"


@data_collect_ex.named_config
def ant():
  env_name = "Ant-v2"
  make_blank_policy_kwargs = dict(
      n_steps=2048,  # batch size of 2048*8=16384 due to num_vec
  )
  total_timesteps = int(5e6)  # OK after 2e6, but continues improving


@data_collect_ex.named_config
def cartpole():
  env_name = "CartPole-v1"
  total_timesteps = int(1e5)


@data_collect_ex.named_config
def half_cheetah():
  env_name = "HalfCheetah-v2"
  total_timesteps = int(5e6)  # does OK after 1e6, but continues improving


@data_collect_ex.named_config
def hopper():
  # TODO(adam): upgrade to Hopper-v3?
  env_name = "Hopper-v2"


@data_collect_ex.named_config
def mountain_car():
  env_name = "MountainCar-v0"


@data_collect_ex.named_config
def humanoid():
  env_name = "Humanoid-v2"
  make_blank_policy_kwargs = dict(
      n_steps=2048,  # batch size of 2048*8=16384 due to num_vec
  )
  total_timesteps = int(10e6)  # fairly discontinuous, needs at least 5e6


@data_collect_ex.named_config
def pendulum():
  env_name = "Pendulum-v0"


@data_collect_ex.named_config
def reacher():
  env_name = "Reacher-v2"


@data_collect_ex.named_config
def swimmer():
  env_name = "Swimmer-v2"


@data_collect_ex.named_config
def walker():
  env_name = "Walker2d-v2"


# Debug configs

@data_collect_ex.named_config
def fast():
  """Intended for testing purposes: small # of updates, ends quickly."""
  total_timesteps = int(1e4)
