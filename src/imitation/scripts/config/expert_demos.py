import os

import sacred

from imitation.scripts.config.common import DEFAULT_INIT_RL_KWARGS
from imitation.util import util

expert_demos_ex = sacred.Experiment("expert_demos")


@expert_demos_ex.config
def expert_demos_defaults():
    env_name = "CartPole-v1"  # The gym.Env name
    total_timesteps = int(1e6)  # Number of training timesteps in model.learn()
    num_vec = 8  # Number of environments in VecEnv
    parallel = True  # Use SubprocVecEnv (generally faster if num_vec>1)
    normalize = True  # Use VecNormalize
    normalize_kwargs = dict()  # kwargs for `VecNormalize`
    max_episode_steps = None  # Set to positive int to limit episode horizons
    n_episodes_eval = 50  # Num of episodes for final ep reward mean evaluation

    init_rl_kwargs = dict(DEFAULT_INIT_RL_KWARGS)

    # If specified, overrides the ground-truth environment reward
    reward_type = None  # override reward type
    reward_path = None  # override reward path

    rollout_save_final = True  # If True, save after training is finished.
    rollout_save_n_timesteps = None  # Min timesteps saved per file, optional.
    rollout_save_n_episodes = None  # Num episodes saved per file, optional.

    policy_save_interval = 10000  # Num timesteps between saves (<=0 disables)
    policy_save_final = True  # If True, save after training is finished.

    init_tensorboard = False  # If True, then write Tensorboard logs.

    log_root = os.path.join("output", "expert_demos")  # output directory


@expert_demos_ex.config
def default_end_cond(rollout_save_n_timesteps, rollout_save_n_episodes):
    # Only set default if both end cond options are None.
    # This way the Sacred CLI caller can set `rollout_save_n_episodes` only
    # without getting an error that `rollout_save_n_timesteps is not None`.
    if rollout_save_n_timesteps is None and rollout_save_n_episodes is None:
        rollout_save_n_timesteps = 2000  # Min timesteps saved per file, optional.


@expert_demos_ex.config
def logging(env_name, log_root):
    log_dir = os.path.join(
        log_root, env_name.replace("/", "_"), util.make_unique_timestamp()
    )


@expert_demos_ex.config
def rollouts_from_policy_only_defaults(log_dir):
    policy_path = None  # Policy path for rollouts_from_policy command only
    policy_type = "ppo"  # Policy type for rollouts_from_policy command only
    rollout_save_path = os.path.join(
        log_dir, "rollout.pkl"
    )  # Save path for `rollouts_from_policy` only.


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
def seals_cartpole():
    env_name = "seals/CartPole-v0"
    total_timesteps = int(1e6)


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
    init_rl_kwargs = dict(
        n_steps=2048,
    )  # batch size of 2048*8=16384 due to num_vec
    total_timesteps = int(10e6)  # fairly discontinuous, needs at least 5e6


@expert_demos_ex.named_config
def mountain_car():
    env_name = "MountainCar-v0"


@expert_demos_ex.named_config
def seals_mountain_car():
    env_name = "seals/MountainCar-v0"


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
    total_timesteps = int(1)
    max_episode_steps = int(1)


# Shared settings

ant_shared_locals = dict(
    init_rl_kwargs=dict(
        n_steps=2048,
    ),  # batch size of 2048*8=16384 due to num_vec
    total_timesteps=int(5e6),
    max_episode_steps=500,  # To match `inverse_rl` settings.
)
