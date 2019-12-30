"""Configuration for imitation.scripts.train_adversarial."""

import os

import sacred
from stable_baselines.common import policies

from imitation import util
from imitation.policies import base
from imitation.scripts.config.common import DEFAULT_INIT_RL_KWARGS

train_ex = sacred.Experiment("train_adversarial", interactive=True)


@train_ex.config
def train_defaults():
  env_name = "CartPole-v1"  # environment to train on

  # Num times to generate a batch and update both generator and discrim.
  # Set `total_timesteps` to None when using `n_epochs`.
  n_epochs = 50

  # Total timesteps taken in env during training.
  # Set `n_epochs` to None when using `total_timesteps`.
  total_timesteps = None

  n_expert_demos = None  # Num demos used. None uses every demo possible
  n_episodes_eval = 50  # Num of episodes for final mean ground truth return
  airl_entropy_weight = 1.0

  # Number of epochs in between plots (<0 disables) (=0 means final plot only)
  plot_interval = -1
  n_plot_episodes = 5  # Number of rollouts for each mean_ep_rew data
  # Interval for extra episode rew data. (<=0 disables)
  extra_episode_data_interval = -1
  show_plots = True  # Show plots in addition to saving them

  init_trainer_kwargs = dict(
      num_vec=8,  # Must evenly divide gen_batch_size
      parallel=True,  # Use SubprocVecEnv (generally faster if num_vec>1)
      max_episode_steps=None,  # Set to positive int to limit episode horizons
      scale=True,

      reward_kwargs=dict(
          theta_units=[32, 32],
          phi_units=[32, 32],
      ),

      init_rl_kwargs=dict(policy_class=base.FeedForward32Policy,
                          **DEFAULT_INIT_RL_KWARGS),
  )

  log_root = os.path.join("output", "train_adversarial")  # output directory
  checkpoint_interval = 5  # num epochs between checkpoints (<=0 disables)
  init_tensorboard = False  # If True, then write Tensorboard logs.
  rollout_hint = None  # Used to generate default rollout_path

  # Batch_size must be a multiple of `init_trainer_kwargs.num_vec`.
  # (If using PPO2, then also must be a multiple of
  # `init_trainer_kwargs.init_rl_kwargs.nminibatch`).
  gen_batch_size = 2048  # Batch size for both generator updates.
  n_disc_minibatch = 4  # Num discriminator minibatches


@train_ex.config
def aliases_default_gen_batch_size(gen_batch_size):
  # Batch size for discriminator updates. Defaults to `gen_batch_size`.
  disc_batch_size = gen_batch_size

  # Setting generator buffer capacity and discriminator batch size to
  # the same number is equivalent to not using a replay buffer at all.
  # "Disabling" the replay buffer seems to improve convergence speed, but may
  # come at a cost of stability.
  gen_replay_buffer_size = gen_batch_size  # Num generator transitions stored


@train_ex.config
def apply_init_trainer_kwargs_aliases(n_disc_minibatch,
                                      disc_batch_size,
                                      gen_replay_buffer_size):
  init_trainer_kwargs = dict(
    trainer_kwargs=dict(
      n_disc_minibatch=n_disc_minibatch,
      gen_replay_buffer_capacity=gen_replay_buffer_size,
      disc_batch_size=disc_batch_size,
    ))


@train_ex.config
def calc_timesteps(n_epochs, total_timesteps, gen_batch_size):
  if total_timesteps is None and n_epochs is not None:
    total_timesteps = n_epochs * gen_batch_size
  else:
    assert total_timesteps is not None and n_epochs is None, (
        "Must set exactly one of n_epochs={} and total_timesteps={} to non-None"
        .format(n_epochs, total_timesteps))


@train_ex.config
def calc_n_steps(init_trainer_kwargs, gen_batch_size):
  _num_vec = init_trainer_kwargs["num_vec"]
  assert gen_batch_size % _num_vec == 0, ("num_vec must evenly divide "
                                          "gen_batch_size")
  init_trainer_kwargs["init_rl_kwargs"]["n_steps"] = gen_batch_size // _num_vec
  del _num_vec


@train_ex.config
def paths(env_name, log_root, rollout_hint):
  log_dir = os.path.join(log_root, env_name.replace('/', '_'),
                         util.make_unique_timestamp())
  # Recommended that user sets rollout_path manually.
  # By default we guess the named config associated with `env_name`
  # and attempt to load rollouts from `data/expert_models/`.
  if rollout_hint is None:
    rollout_hint = env_name.split("-")[0].lower()
  rollout_path = os.path.join("data/expert_models",
                              f"{rollout_hint}_0",
                              "rollouts", "final.pkl")


# Training algorithm named configs

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


# Standard Gym environment named configs

@train_ex.named_config
def acrobot():
  env_name = "Acrobot-v1"
  rollout_hint = "acrobot"


@train_ex.named_config
def ant():
  env_name = "Ant-v2"
  rollout_hint = "ant"
  locals().update(**ant_shared_locals)


@train_ex.named_config
def cartpole():
  env_name = "CartPole-v1"
  rollout_hint = "cartpole"
  init_trainer_kwargs = dict(
      scale=False,
  )


@train_ex.named_config
def half_cheetah():
  env_name = "HalfCheetah-v2"
  rollout_hint = "half_cheetah"
  n_epochs = 1000

  init_trainer_kwargs = dict(
      airl_entropy_weight=0.1,
  )


@train_ex.named_config
def hopper():
  # TODO(adam): upgrade to Hopper-v3?
  env_name = "Hopper-v2"
  rollout_hint = "hopper"


@train_ex.named_config
def humanoid():
  env_name = "Humanoid-v2"
  rollout_hint = "humanoid"
  n_epochs = 2000


@train_ex.named_config
def mountain_car():
  env_name = "MountainCar-v0"
  rollout_hint = "mountain_car"


@train_ex.named_config
def pendulum():
  env_name = "Pendulum-v0"
  rollout_hint = "pendulum"


@train_ex.named_config
def reacher():
  env_name = "Reacher-v2"
  rollout_hint = "reacher"


@train_ex.named_config
def swimmer():
  env_name = "Swimmer-v2"
  rollout_hint = "swimmer"
  n_epochs = 1000
  init_trainer_kwargs = dict(
      init_rl_kwargs=dict(
          policy_network_class=policies.MlpPolicy,
      ),
  )


@train_ex.named_config
def walker():
  env_name = "Walker2d-v2"
  rollout_hint = "walker"


# Custom Gym environment named configs

@train_ex.named_config
def two_d_maze():
  env_name = "imitation/TwoDMaze-v0"
  rollout_hint = "two_d_maze"


@train_ex.named_config
def custom_ant():
  env_name = "imitation/CustomAnt-v0"
  rollout_hint = "custom_ant"
  locals().update(**ant_shared_locals)


@train_ex.named_config
def disabled_ant():
  env_name = "imitation/DisabledAnt-v0"
  rollout_hint = "disabled_ant"
  locals().update(**ant_shared_locals)


# Debug configs

@train_ex.named_config
def fast():
  """Minimize the amount of computation. Useful for test cases."""
  n_epochs = 1
  n_expert_demos = 1
  n_episodes_eval = 1
  gen_batch_size = 2
  show_plots = False
  n_plot_episodes = 1
  init_trainer_kwargs = dict(
    parallel=False,  # easier to debug with everything in one process
    max_episode_steps=int(1e2),
    num_vec=2,
    init_rl_kwargs=dict(nminibatches=1),
  )


# Shared settings

ant_shared_locals = dict(
    n_epochs=2000,
    gen_batch_size=2048*8,
    init_trainer_kwargs=dict(
        max_episode_steps=500,  # To match `inverse_rl` settings.
    ),
)
