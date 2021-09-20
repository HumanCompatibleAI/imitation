"""Configuration settings for train_dagger, training DAgger from synthetic demos."""

import os

import sacred
import torch as th

from imitation.scripts.common import train as train_common

train_dagger_ex = sacred.Experiment(
    "train_dagger",
    ingredients=[train_common.train_ingredient],
)


@train_dagger_ex.config
def config():
    use_offline_rollouts = False
    # TODO(shwang): This config is almost the same as train_bc's. Consider merging
    #   into a shared ingredient.
    bc_train_kwargs = dict(
        n_epochs=None,  # Number of BC epochs per DAgger training round
        n_batches=None,  # Number of BC batches per DAgger training round
        log_interval=500,  # Number of updates between Tensorboard/stdout logs
    )

    batch_size = 32

    l2_weight = 3e-5  # L2 regularization weight
    # Path to directory containing model.pkl (and optionally, vec_normalize.pkl)
    expert_policy_path = None
    expert_policy_type = None  # 'ppo', 'random', or 'zero'

    total_timesteps = 1e5

    optimizer_cls = th.optim.Adam
    optimizer_kwargs = dict(
        lr=4e-4,
    )


@train_dagger_ex.config
def defaults(
    train,
    expert_policy_type,
    expert_policy_path,
):
    if expert_policy_type is None and expert_policy_path is None:
        expert_policy_type = "ppo"
        expert_policy_path = os.path.join(
            train_common.guess_expert_dir(train["data_dir"], train["env_name"]),
            "policies",
            "final",
        )


@train_dagger_ex.config
def default_train_duration(bc_train_kwargs):
    if (
        bc_train_kwargs.get("n_epochs") is None
        and bc_train_kwargs.get("n_batches") is None
    ):
        bc_train_kwargs["n_epochs"] = 4


# TODO(shwang): Move these redundant configs into a `auto.env` Ingredient,
# similar to what the ILR project does.
@train_dagger_ex.named_config
def mountain_car():
    train = dict(env_name="MountainCar-v0")
    l2_weight = 0
    total_timesteps = 20000


@train_dagger_ex.named_config
def seals_mountain_car():
    train = dict(env_name="seals/MountainCar-v0")
    l2_weight = 0
    total_timesteps = 20000


@train_dagger_ex.named_config
def cartpole():
    train = dict(env_name="CartPole-v1")
    total_timesteps = 20000


@train_dagger_ex.named_config
def seals_cartpole():
    train = dict(env_name="seals/CartPole-v0")
    total_timesteps = 20000


@train_dagger_ex.named_config
def pendulum():
    train = dict(env_name="Pendulum-v0")


@train_dagger_ex.named_config
def ant():
    train = dict(env_name="Ant-v2")


@train_dagger_ex.named_config
def half_cheetah():
    train = dict(env_name="HalfCheetah-v2")
    l2_weight = 0
    total_timesteps = 60000


@train_dagger_ex.named_config
def humanoid():
    train = dict(env_name="Humanoid-v2")


@train_dagger_ex.named_config
def fast():
    total_timesteps = 50
    bc_train_kwargs = dict(
        n_batches=50,
    )
