"""Configuration settings for train_rl, training a policy with RL."""


import sacred
from torch import nn

from imitation.scripts.ingredients import environment
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation, rl

# Note: All the hyperparameter configs in the file are tuned
# for the PPO algorithm on the respective environment using the
# RL Baselines Zoo library:
# https://github.com/HumanCompatibleAI/rl-baselines3-zoo/

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
    environment = dict(gym_id="seals/CartPole-v0", num_vec=8)
    total_timesteps = int(1e5)
    policy = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        ),
    )
    normalize_reward = False
    rl = dict(
        batch_size=4096,
        rl_kwargs=dict(
            batch_size=256,
            clip_range=0.4,
            ent_coef=0.008508727919228772,
            gae_lambda=0.9,
            gamma=0.9999,
            learning_rate=0.0012403278189645594,
            max_grad_norm=0.8,
            n_epochs=10,
            vf_coef=0.489343896591493,
        ),
    )


@train_rl_ex.named_config
def half_cheetah():
    environment = dict(gym_id="HalfCheetah-v3")
    total_timesteps = int(5e6)  # does OK after 1e6, but continues improving


@train_rl_ex.named_config
def seals_half_cheetah():
    environment = dict(
        gym_id="seals/HalfCheetah-v0",
        num_vec=1,
    )

    policy = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.Tanh,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        ),
    )
    # total_timesteps = int(5e6)  # does OK after 1e6, but continues improving
    total_timesteps = 1e6
    normalize_reward = False

    rl = dict(
        batch_size=512,
        rl_kwargs=dict(
            batch_size=64,
            clip_range=0.1,
            ent_coef=3.794797423594763e-06,
            gae_lambda=0.95,
            gamma=0.95,
            learning_rate=0.0003286871805949382,
            max_grad_norm=0.8,
            n_epochs=5,
            vf_coef=0.11483689492120866,
        ),
    )


@train_rl_ex.named_config
def seals_hopper():
    environment = dict(gym_id="seals/Hopper-v0", num_vec=1)
    policy = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        ),
    )

    total_timesteps = 1e6
    normalize_reward = False

    rl = dict(
        batch_size=2048,
        rl_kwargs=dict(
            batch_size=512,
            clip_range=0.1,
            ent_coef=0.0010159833764878474,
            gae_lambda=0.98,
            gamma=0.995,
            learning_rate=0.0003904770450788824,
            max_grad_norm=0.9,
            n_epochs=20,
            # policy_kwargs are same as the defaults
            vf_coef=0.20315938606555833,
        ),
    )


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
    environment = dict(
        gym_id="seals/Ant-v0",
        num_vec=1,
    )

    policy = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.Tanh,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        ),
    )

    total_timesteps = 1e6
    normalize_reward = False

    rl = dict(
        batch_size=2048,
        rl_kwargs=dict(
            batch_size=16,
            clip_range=0.3,
            ent_coef=3.1441389214159857e-06,
            gae_lambda=0.8,
            gamma=0.995,
            learning_rate=0.00017959211641976886,
            max_grad_norm=0.9,
            n_epochs=10,
            # policy_kwargs are same as the defaults
            vf_coef=0.4351450387648799,
        ),
    )


@train_rl_ex.named_config
def seals_swimmer():
    environment = dict(gym_id="seals/Swimmer-v0", num_vec=1)
    policy = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        ),
    )

    total_timesteps = 1e6
    normalize_reward = False

    rl = dict(
        batch_size=2048,
        rl_kwargs=dict(
            batch_size=64,
            clip_range=0.1,
            ent_coef=5.167107294612664e-08,
            gae_lambda=0.95,
            gamma=0.999,
            learning_rate=0.000414936134792374,
            max_grad_norm=2,
            n_epochs=5,
            # policy_kwargs are same as the defaults
            vf_coef=0.6162112311062333,
        ),
    )


@train_rl_ex.named_config
def seals_walker():
    environment = dict(gym_id="seals/Walker2d-v0", num_vec=1)
    policy = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        ),
    )

    total_timesteps = 1e6
    normalize_reward = False

    rl = dict(
        batch_size=8192,
        rl_kwargs=dict(
            batch_size=128,
            clip_range=0.4,
            ent_coef=0.00013057334805552262,
            gae_lambda=0.92,
            gamma=0.98,
            learning_rate=0.000138575372312869,
            max_grad_norm=0.6,
            n_epochs=20,
            # policy_kwargs are same as the defaults
            vf_coef=0.6167177795726859,
        ),
    )


# Debug configs


@train_rl_ex.named_config
def fast():
    # Intended for testing purposes: small # of updates, ends quickly.
    total_timesteps = int(4)
    policy_save_interval = 2
