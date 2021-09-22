"""Configuration settings for train_dagger, training DAgger from synthetic demos."""

import os

import sacred
import torch as th
from stable_baselines3.common import utils

from imitation.policies import base
from imitation.scripts.common import train as train_common

train_imitation_ex = sacred.Experiment(
    "train_imitation",
    ingredients=[train_common.train_ingredient],
)


@train_imitation_ex.config
def config():
    policy_cls = base.FeedForward32Policy
    policy_kwargs = {
        # parameter mandatory for ActorCriticPolicy, but not used by BC
        "lr_schedule": utils.get_schedule_fn(1),
    }
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
        expert_policy_path=None,  # path to directory containing model.pkl
        expert_policy_type=None,  # 'ppo', 'random', or 'zero'
        total_timesteps=1e5,
    )


@train_imitation_ex.config
def defaults(
    train,
    dagger,
):
    if dagger["expert_policy_type"] is None and dagger["expert_policy_path"] is None:
        dagger = dict(
            expert_policy_type="ppo",
            expert_policy_path=os.path.join(
                train_common.guess_expert_dir(train["data_dir"], train["env_name"]),
                "policies",
                "final",
            ),
        )


# TODO(shwang): Move these redundant configs into a `auto.env` Ingredient,
# similar to what the ILR project does.
@train_imitation_ex.named_config
def mountain_car():
    train = dict(env_name="MountainCar-v0")
    bc_kwargs = dict(l2_weight=0.0)
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def seals_mountain_car():
    train = dict(env_name="seals/MountainCar-v0")
    bc_kwargs = dict(l2_weight=0.0)
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def cartpole():
    train = dict(env_name="CartPole-v1")
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def seals_cartpole():
    train = dict(env_name="seals/CartPole-v0")
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def pendulum():
    train = dict(env_name="Pendulum-v0")


@train_imitation_ex.named_config
def ant():
    train = dict(env_name="Ant-v2")


@train_imitation_ex.named_config
def seals_ant():
    train = dict(env_name="seals/Ant-v0")


@train_imitation_ex.named_config
def half_cheetah():
    train = dict(env_name="HalfCheetah-v2")
    bc_kwargs = dict(l2_weight=0.0)
    dagger = dict(total_timesteps=60000)


@train_imitation_ex.named_config
def humanoid():
    train = dict(env_name="Humanoid-v2")


@train_imitation_ex.named_config
def seals_humanoid():
    train = dict(env_name="seals/Humanoid-v0")


@train_imitation_ex.named_config
def fast():
    dagger = dict(total_timesteps=50)
    bc_train_kwargs = dict(n_batches=50)
