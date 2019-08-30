import os

import sacred

from imitation.scripts.config.common import DEFAULT_BLANK_POLICY_KWARGS
from imitation.util import util

expert_demos_ex = sacred.Experiment("expert_demos")


@expert_demos_ex.config
def expert_demos_defaults():
  env_name = "CartPole-v1"  # The gym.Env name
  total_timesteps = int(1e6)  # Number of training timesteps in model.learn()
  num_vec = 8  # Number of environments in VecEnv
  parallel = True  # Use SubprocVecEnv (generally faster if num_vec>1)
  normalize = True  # Use VecNormalize
  max_episode_steps = None  # Set to positive int to limit episode horizons
  make_blank_policy_kwargs = dict(DEFAULT_BLANK_POLICY_KWARGS)

  # If specified, overrides the ground-truth environment reward
  reward_type = None  # override reward type
  reward_path = None   # override reward path

  rollout_save_interval = -1  # Num updates between saves (<=0 disables)
  rollout_save_final = True  # If True, save after training is finished.
  rollout_save_n_timesteps = 2000  # Min timesteps saved per file, optional.
  rollout_save_n_episodes = None  # Num episodes saved per file, optional.

  ray_tune_interval = -1  # Num updates between `ray.track.log`. (<=0 disables)

  policy_save_interval = 100  # Num updates between saves (<=0 disables)
  policy_save_final = True  # If True, save after training is finished.

  log_root = os.path.join("output", "expert_demos")  # output directory


@expert_demos_ex.config
def logging(env_name, log_root):
  log_dir = os.path.join(log_root, env_name.replace('/', '_'),
                         util.make_unique_timestamp())


@expert_demos_ex.named_config
def ray_tune():
  ray_tune_interval = 100


# Standard Gym env configs

@expert_demos_ex.named_config
def acrobot():
  env_name = "Acrobot-v1"


@expert_demos_ex.named_config
def ant():
  env_name = "Ant-v2"
  locals().update(**ant_shared_locals)


@expert_demos_ex.named_config
def cartpole():
  env_name = "CartPole-v1"
  total_timesteps = int(1e5)


@expert_demos_ex.named_config
def half_cheetah():
  env_name = "HalfCheetah-v2"
  total_timesteps = int(5e6)  # does OK after 1e6, but continues improving


@expert_demos_ex.named_config
def hopper():
  # TODO(adam): upgrade to Hopper-v3?
  env_name = "Hopper-v2"


@expert_demos_ex.named_config
def humanoid():
  env_name = "Humanoid-v2"
  make_blank_policy_kwargs = dict(
      n_steps=2048,  # batch size of 2048*8=16384 due to num_vec
  )
  total_timesteps = int(10e6)  # fairly discontinuous, needs at least 5e6


@expert_demos_ex.named_config
def mountain_car():
  env_name = "MountainCar-v0"


@expert_demos_ex.named_config
def pendulum():
  env_name = "Pendulum-v0"


@expert_demos_ex.named_config
def reacher():
  env_name = "Reacher-v2"


@expert_demos_ex.named_config
def swimmer():
  env_name = "Swimmer-v2"


@expert_demos_ex.named_config
def walker():
  env_name = "Walker2d-v2"


# Custom env configs

@expert_demos_ex.named_config
def custom_ant():
  env_name = "imitation/CustomAnt-v0"
  locals().update(**ant_shared_locals)


@expert_demos_ex.named_config
def disabled_ant():
  env_name = "imitation/DisabledAnt-v0"
  locals().update(**ant_shared_locals)


@expert_demos_ex.named_config
def two_d_maze():
  env_name = "imitation/TwoDMaze-v0"


# Debug configs

@expert_demos_ex.named_config
def fast():
  """Intended for testing purposes: small # of updates, ends quickly."""
  total_timesteps = int(1e3)
  max_episode_steps = int(1e3)


# Shared settings

ant_shared_locals = dict(
    make_blank_policy_kwargs=dict(
        n_steps=2048,  # batch size of 2048*8=16384 due to num_vec
    ),
    total_timesteps=int(5e6),
    max_episode_steps=500,  # To match `inverse_rl` settings.
)
