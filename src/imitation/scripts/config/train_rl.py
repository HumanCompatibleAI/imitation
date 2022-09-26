"""Configuration settings for train_rl, training a policy with RL."""


import sacred
from torch import nn as nn

from imitation.scripts.common import common, rl, train

train_rl_ex = sacred.Experiment(
    "train_rl",
    ingredients=[common.common_ingredient, train.train_ingredient, rl.rl_ingredient],
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
    common = dict(env_name="Acrobot-v1")


@train_rl_ex.named_config
def ant():
    common = dict(env_name="Ant-v2")
    rl = dict(batch_size=16384)
    total_timesteps = int(5e6)


@train_rl_ex.named_config
def cartpole():
    common = dict(env_name="CartPole-v1")
    total_timesteps = int(1e5)


@train_rl_ex.named_config
def seals_cartpole():
    common = dict(env_name="seals/CartPole-v0")
    total_timesteps = int(1e6)


@train_rl_ex.named_config
def half_cheetah():
    common = dict(env_name="HalfCheetah-v3")
    total_timesteps = int(5e6)  # does OK after 1e6, but continues improving


@train_rl_ex.named_config
def seals_half_cheetah():
    common = dict(
        env_name="seals/HalfCheetah-v0",
        num_vec=1,
    )

    train = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.Tanh,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        ),
    )
    # total_timesteps = int(5e6)  # does OK after 1e6, but continues improving
    total_timesteps = 1e6
    normalize_reward = True

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
    common = dict(env_name="seals/Hopper-v0")


@train_rl_ex.named_config
def seals_humanoid():
    common = dict(env_name="seals/Humanoid-v0")
    rl = dict(batch_size=16384)
    total_timesteps = int(10e6)  # fairly discontinuous, needs at least 5e6


@train_rl_ex.named_config
def mountain_car():
    common = dict(env_name="MountainCar-v0")


@train_rl_ex.named_config
def seals_mountain_car():
    common = dict(env_name="seals/MountainCar-v0")


@train_rl_ex.named_config
def pendulum():
    common = dict(env_name="Pendulum-v1", num_vec=4)
    total_timesteps = int(1e5)

    train = dict(
        policy_cls="MlpPolicy",
        # policy_kwargs=dict(
        #     activation_fn=nn.Tanh,
        #     net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        # ),
    )
    normalize_reward = False

    rl = dict(
        batch_size=1024 * 4,
        rl_kwargs=dict(
            gae_lambda=0.95,
            gamma=0.9,
            n_epochs=10,
            ent_coef=0.0,
            learning_rate=1e-3,
            clip_range=0.2,
            use_sde=True,
            sde_sample_freq=4,
            # batch_size=64,
            # max_grad_norm=0.8,
            # vf_coef=0.11483689492120866,
        ),
    )


@train_rl_ex.named_config
def reacher():
    common = dict(env_name="Reacher-v2")


@train_rl_ex.named_config
def seals_ant():
    common = dict(
        env_name="seals/Ant-v0",
        num_vec=1,
    )

    train = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.Tanh,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        ),
    )

    total_timesteps = 1e6
    normalize_reward = True

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
    common = dict(env_name="seals/Swimmer-v0", num_vec=1)
    train = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.Tanh,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        ),
    )

    total_timesteps = 1e6
    normalize_reward = True

    rl = dict(
        batch_size=2048,
        rl_kwargs=dict(
            batch_size=8,
            clip_range=0.1,
            ent_coef=5.167107294612664e-08,
            gae_lambda=0.95,
            gamma=0.999,
            learning_rate=0.0001214437022727675,
            max_grad_norm=2,
            n_epochs=20,
            # policy_kwargs are same as the defaults
            vf_coef=0.6162112311062333,
        ),
    )


@train_rl_ex.named_config
def seals_walker():
    common = dict(env_name="seals/Walker2d-v0", num_vec=1)
    train = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        ),
    )

    total_timesteps = 1e6
    normalize_reward = True

    rl = dict(
        batch_size=2048,
        rl_kwargs=dict(
            batch_size=8,
            clip_range=0.4,
            ent_coef=0.00013057334805552262,
            gae_lambda=0.92,
            gamma=0.98,
            learning_rate=3.791707778339674e-05,
            max_grad_norm=0.6,
            n_epochs=5,
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
