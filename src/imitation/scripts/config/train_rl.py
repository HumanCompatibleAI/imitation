"""Configuration settings for train_rl, training a policy with RL."""

import sacred

from imitation.scripts.ingredients import environment
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation, rl

train_rl_ex = sacred.Experiment(
    "train_rl",
    ingredients=[
        logging_ingredient.logging_ingredient,
        environment.environment_ingredient,
        rl.rl_ingredient,
        policy_evaluation.policy_evaluation_ingredient,
    ],
)


@train_rl_ex.config
def train_rl_defaults():
    total_timesteps = int(1e6)  # Number of training timesteps in model.learn()
    normalize_reward = True  # Use VecNormalize to normalize the reward
    normalize_kwargs = dict()  # kwargs for `VecNormalize`

    # If specified, overrides the ground-truth environment reward
    reward_type = None  # override reward type
    reward_path = None  # override reward path
    load_reward_kwargs = {}

    rollout_save_final = True  # If True, save after training is finished.
    rollout_save_n_timesteps = None  # Min timesteps saved per file, optional.
    rollout_save_n_episodes = None  # Num episodes saved per file, optional.

    policy_save_interval = 10000  # Num timesteps between saves (<=0 disables)
    policy_save_final = True  # If True, save after training is finished.

    agent_path = None  # Path to load agent from, optional.


@train_rl_ex.config
def default_end_cond(rollout_save_n_timesteps, rollout_save_n_episodes):
    # Only set default if both end cond options are None.
    # This way the Sacred CLI caller can set `rollout_save_n_episodes` only
    # without getting an error that `rollout_save_n_timesteps is not None`.
    if rollout_save_n_timesteps is None and rollout_save_n_episodes is None:
        rollout_save_n_timesteps = 2000  # Min timesteps saved per file, optional.


# Standard Gym env configs


@train_rl_ex.named_config
def acrobot():
    environment = dict(gym_id="Acrobot-v1")


@train_rl_ex.named_config
def ant():
    environment = dict(gym_id="Ant-v2")
    rl = dict(batch_size=16384)
    total_timesteps = int(5e6)


@train_rl_ex.named_config
def cartpole():
    environment = dict(gym_id="CartPole-v1")
    total_timesteps = int(1e5)


@train_rl_ex.named_config
def seals_cartpole():
    environment = dict(gym_id="seals/CartPole-v0")
    total_timesteps = int(1e6)


@train_rl_ex.named_config
def half_cheetah():
    environment = dict(gym_id="HalfCheetah-v3")
    total_timesteps = int(5e6)  # does OK after 1e6, but continues improving


@train_rl_ex.named_config
def seals_hopper():
    environment = dict(gym_id="seals/Hopper-v0")


@train_rl_ex.named_config
def seals_humanoid():
    environment = dict(gym_id="seals/Humanoid-v0")
    rl = dict(batch_size=16384)
    total_timesteps = int(10e6)  # fairly discontinuous, needs at least 5e6


@train_rl_ex.named_config
def mountain_car():
    environment = dict(gym_id="MountainCar-v0")


@train_rl_ex.named_config
def seals_mountain_car():
    environment = dict(gym_id="seals/MountainCar-v0")


@train_rl_ex.named_config
def pendulum():
    environment = dict(gym_id="Pendulum-v1")
    rl = dict(
        batch_size=4096,
        rl_kwargs=dict(
            gamma=0.9,
            learning_rate=1e-3,
        ),
    )
    total_timesteps = int(2e5)


@train_rl_ex.named_config
def reacher():
    environment = dict(gym_id="Reacher-v2")


@train_rl_ex.named_config
def seals_ant():
    environment = dict(gym_id="seals/Ant-v0")


@train_rl_ex.named_config
def seals_swimmer():
    environment = dict(gym_id="seals/Swimmer-v0")


@train_rl_ex.named_config
def seals_walker():
    environment = dict(gym_id="seals/Walker2d-v0")


# Debug configs


@train_rl_ex.named_config
def fast():
    # Intended for testing purposes: small # of updates, ends quickly.
    total_timesteps = int(4)
    policy_save_interval = 2
