"""Configuration settings for train_bc, training a policy via behavioral cloning."""

import sacred
import torch as th
from stable_baselines3.common import utils

from imitation.policies import base
from imitation.scripts.common import train

train_bc_ex = sacred.Experiment("train_bc", ingredients=[train.train_ingredient])


@train_bc_ex.config
def config():
    n_epochs = None  # Number of training epochs (mutually exclusive with n_batches)
    n_batches = None  # Number of training batches (mutually exclusive with n_epochs)

    policy_cls = base.FeedForward32Policy
    policy_kwargs = {
        # parameter mandatory for ActorCriticPolicy, but not used by BC
        "lr_schedule": utils.get_schedule_fn(1),
    }

    bc_kwargs = dict(
        demo_batch_size=32,
        l2_weight=3e-5,  # L2 regularization weight
        optimizer_cls=th.optim.Adam,
        optimizer_kwargs=dict(
            lr=4e-4,
        ),
    )

    log_interval = 1000  # Number of batches in between each training log.
    log_rollouts_n_episodes = 5  # Number of rollout episodes per training log.


@train_bc_ex.config
def default_train_duration(n_epochs, n_batches):
    if n_epochs is None and n_batches is None:
        n_batches = 50_000


# TODO(adam): add more environment configs?
# (many MuJoCo ones are missing...)
@train_bc_ex.named_config
def mountain_car():
    train = dict(env_name="MountainCar-v0")


@train_bc_ex.named_config
def seals_mountain_car():
    train = dict(env_name="seals/MountainCar-v0")


@train_bc_ex.named_config
def cartpole():
    train = dict(env_name="CartPole-v1")


@train_bc_ex.named_config
def seals_cartpole():
    train = dict(env_name="seals/CartPole-v0")


@train_bc_ex.named_config
def pendulum():
    train = dict(env_name="Pendulum-v0")


@train_bc_ex.named_config
def seals_ant():
    train = dict(env_name="seals/Ant-v0")


@train_bc_ex.named_config
def half_cheetah():
    train = dict(env_name="HalfCheetah-v2")
    n_batches = 100_000


@train_bc_ex.named_config
def seals_humanoid():
    train = dict(env_name="seals/Humanoid-v0")


@train_bc_ex.named_config
def fast():
    n_batches = 50
