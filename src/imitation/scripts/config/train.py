"""Configuration for imitation.scripts.train."""

import os

import sacred
from stable_baselines.common import policies

from imitation import util
from imitation.policies import base
from imitation.scripts.config.common import DEFAULT_BLANK_POLICY_KWARGS

train_ex = sacred.Experiment("train", interactive=True)


@train_ex.config
def train_defaults():
    env_name = "CartPole-v1"  # environment to train on
    n_epochs = 50
    n_episodes_eval = 50  # Num of episodes for final mean ground truth return
    n_disc_steps_per_epoch = 50
    n_gen_steps_per_epoch = 2048
    use_gail = True
    airl_entropy_weight = 1.0

    init_trainer_kwargs = dict(
        num_vec=8,  # NOTE: changing this also changes the effective n_steps!
        parallel=True,  # Use SubprocVecEnv (generally faster if num_vec>1)
        scale=True,

        reward_kwargs=dict(
            theta_units=[32, 32],
            phi_units=[32, 32],
        ),

        trainer_kwargs=dict(
            n_disc_samples_per_buffer=1000,
            # Setting buffer capacity and disc samples to 1000 effectively
            # disables the replay buffer. This seems to improve convergence
            # speed, but may come at a cost of stability.
            gen_replay_buffer_capacity=1000,
        ),

        make_blank_policy_kwargs=dict(policy_class=base.FeedForward32Policy,
                                      **DEFAULT_BLANK_POLICY_KWARGS),
    )

    log_root = os.path.join("output", "train")  # output directory
    checkpoint_interval = 5  # num epochs between checkpoints (<=0 disables)


@train_ex.config
def paths(env_name, log_root):
    log_dir = os.path.join(log_root, env_name.replace('/', '_'),
                           util.make_unique_timestamp())
    # Recommended user sets rollout_glob manually
    rollout_glob = os.path.join("output", "data_collect",
                                env_name.replace('/', '_'),
                                "*", "rollouts", "final.pkl")


@train_ex.named_config
def gail():
    init_trainer_kwargs = dict(
        use_gail=True,
    )


@train_ex.named_config
def airl():
    init_trainer_kwargs = dict(
        use_gail=False,
    )


@train_ex.named_config
def ant():
    env_name = "Ant-v2"
    n_epochs = 2000


@train_ex.named_config
def acrobot():
  env_name = "Acrobot-v1"


@train_ex.named_config
def cartpole():
    env_name = "CartPole-v1"
    init_trainer_kwargs = dict(
        scale=False,
    )


@train_ex.named_config
def half_cheetah():
    env_name = "HalfCheetah-v2"
    n_epochs = 1000

    init_trainer_kwargs = dict(
        airl_entropy_weight=0.1,
    )


@train_ex.named_config
def hopper():
  # TODO(adam): upgrade to Hopper-v3?
  env_name = "Hopper-v2"


@train_ex.named_config
def humanoid():
  env_name = "Humanoid-v2"
  n_epochs = 2000


@train_ex.named_config
def mountain_car():
  env_name = "MountainCar-v0"


@train_ex.named_config
def pendulum():
    env_name = "Pendulum-v0"


@train_ex.named_config
def reacher():
  env_name = "Reacher-v2"


@train_ex.named_config
def walker():
  env_name = "Walker2d-v2"


@train_ex.named_config
def swimmer():
  env_name = "Swimmer-v2"
  n_epochs = 1000
  init_trainer_kwargs = dict(
      make_blank_policy_kwargs=dict(
          policy_network_class=policies.MlpPolicy,
      ),
  )


@train_ex.named_config
def fast():
  """Minimize the amount of computation. Useful for test cases."""
  n_epochs = 1
  n_episodes_eval = 1
  interactive = False
  n_disc_steps_per_epoch = 1
  n_gen_steps_per_epoch = 1
  n_episodes_per_reward_data = 1
  init_trainer_kwargs = dict(
      n_expert_demos=1,
      parallel=False,  # easier to debug with everything in one process
  )
