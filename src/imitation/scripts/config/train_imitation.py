"""Configuration settings for train_dagger, training DAgger from synthetic demos."""

import sacred
import torch as th
from gym.wrappers import TimeLimit
from seals.util import AutoResetWrapper
from stable_baselines3.common import torch_layers
from stable_baselines3.common.atari_wrappers import AtariWrapper

from imitation.policies import base
from imitation.scripts.common import common
from imitation.scripts.common import demonstrations as demos_common
from imitation.scripts.common import expert, train

train_imitation_ex = sacred.Experiment(
    "train_imitation",
    ingredients=[
        common.common_ingredient,
        demos_common.demonstrations_ingredient,
        train.train_ingredient,
        expert.expert_ingredient,
    ],
)


@train_imitation_ex.config
def config():
    bc_kwargs = dict(
        batch_size=32,
        l2_weight=3e-5,  # L2 regularization weight
        optimizer_cls=th.optim.Adam,
        optimizer_kwargs=dict(
            lr=4e-4,
        ),
    )
    bc_train_kwargs = dict(
        n_epochs=None,  # Number of BC epochs per DAgger training round
        n_batches=None,  # Number of BC batches per DAgger training round
        log_interval=500,  # Number of updates between Tensorboard/stdout logs
    )
    dagger = dict(
        use_offline_rollouts=False,  # warm-start policy with BC from offline demos
        total_timesteps=1e5,
    )
    agent_path = None  # Path to load agent from, optional.


@train_imitation_ex.named_config
def mountain_car():
    common = dict(env_name="MountainCar-v0")
    bc_kwargs = dict(l2_weight=0.0)
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def seals_mountain_car():
    common = dict(env_name="seals/MountainCar-v0")
    bc_kwargs = dict(l2_weight=0.0)
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def cartpole():
    common = dict(env_name="CartPole-v1")
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def seals_cartpole():
    common = dict(env_name="seals/CartPole-v0")
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def pendulum():
    common = dict(env_name="Pendulum-v1")


@train_imitation_ex.named_config
def ant():
    common = dict(env_name="Ant-v2")


@train_imitation_ex.named_config
def seals_ant():
    common = dict(env_name="seals/Ant-v0")


@train_imitation_ex.named_config
def half_cheetah():
    common = dict(env_name="HalfCheetah-v2")
    bc_kwargs = dict(l2_weight=0.0)
    dagger = dict(total_timesteps=60000)


@train_imitation_ex.named_config
def seals_half_cheetah():
    common = dict(env_name="seals/HalfCheetah-v0")
    bc_kwargs = dict(l2_weight=0.0)
    dagger = dict(total_timesteps=60000)


@train_imitation_ex.named_config
def humanoid():
    common = dict(env_name="Humanoid-v2")


@train_imitation_ex.named_config
def seals_humanoid():
    common = dict(env_name="seals/Humanoid-v0")


@train_imitation_ex.named_config
def asteroids():
    common = dict(
        env_name="AsteroidsNoFrameskip-v4",
        post_wrappers=[
            lambda env, _: AutoResetWrapper(env),
            lambda env, _: AtariWrapper(env, terminal_on_life_loss=False),
            lambda env, _: TimeLimit(env, max_episode_steps=100_000),
        ],
    )
    train = dict(
        policy_kwargs=dict(transpose_input=True),
    )
    transpose_obs = True


@train_imitation_ex.named_config
def asteroids_short_episodes():
    common = dict(
        env_name="AsteroidsNoFrameskip-v4",
        post_wrappers=[
            lambda env, _: AutoResetWrapper(env),
            lambda env, _: AtariWrapper(env, terminal_on_life_loss=False),
            lambda env, _: TimeLimit(env, max_episode_steps=100),
        ],
    )
    train = dict(
        policy_kwargs=dict(transpose_input=True),
    )
    transpose_obs = True


@train_imitation_ex.named_config
def cnn():
    train = dict(
        policy_cls=base.CnnPolicy,
        policy_kwargs=dict(features_extractor_class=torch_layers.NatureCNN),
    )


@train_imitation_ex.named_config
def fast():
    dagger = dict(total_timesteps=50)
    bc_train_kwargs = dict(n_batches=50)
