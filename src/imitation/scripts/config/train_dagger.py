"""Configuration settings for train_dagger, training DAgger from synthetic demos."""

import pathlib

import sacred
import torch as th

from imitation.util import util

train_dagger_ex = sacred.Experiment("train_dagger")


@train_dagger_ex.config
def config():
    # TODO(shwang): This config is almost the same as train_bc's. Consider merging
    #   into a shared ingredient.
    env_name = "CartPole-v1"
    venv = None
    bc_train_kwargs = dict(
        n_epochs=None,  # Number of BC epochs per DAgger training round
        n_batches=None,  # Number of BC batches per DAgger training round
        log_interval=500,  # Number of updates between Tensorboard/stdout logs
    )
    n_episodes_eval = 50  # Number of rollout episodes in final evaluation.
    expert_data_src = None
    expert_data_src_format = None  # Either "path" or "trajectory" or None
    rollout_hint = None  # Used to generate default `expert_data_src`.
    # Number of trajectories to use during training, or None to use all.
    n_expert_demos = None

    batch_size = 32
    observation_space = None
    action_space = None

    l2_weight = 3e-5  # L2 regularization weight
    # Path to directory containing model.pkl (and optionally, vec_normalize.pkl)
    expert_policy_path = None
    expert_policy_type = None  # 'ppo', 'random', or 'zero'

    total_timesteps = 1e5

    log_root = pathlib.Path("output", "train_dagger")  # output directory
    optimizer_cls = th.optim.Adam
    optimizer_kwargs = dict(
        lr=4e-4,
    )


@train_dagger_ex.config
def defaults(
    env_name,
    venv,
    rollout_hint,
    expert_data_src,
    expert_data_src_format,
    expert_policy_path,
):
    if expert_data_src is None and expert_data_src_format is None:
        expert_data_src = (
            f"data/expert_models/{rollout_hint or 'cartpole'}_0/rollouts/final.pkl"
        )
        expert_data_src_format = "path"

    if expert_data_src_format is None:
        expert_data_src_format = "path"

    if expert_policy_path is None and expert_policy_path is None:
        expert_policy_path = (
            f"data/expert_models/{rollout_hint or 'cartpole'}_0/policies/final"
        )
        expert_policy_type = "ppo"

    if env_name is not None and venv is None:
        venv = util.make_vec_env(env_name)  # Automatically initialized using `env_name`


@train_dagger_ex.config
def default_train_duration(bc_train_kwargs):
    if (
        bc_train_kwargs.get("n_epochs") is None
        and bc_train_kwargs.get("n_batches") is None
    ):
        bc_train_kwargs["n_epochs"] = 4


@train_dagger_ex.config
def paths(log_root, env_name):
    if env_name is None:
        _env_name_part = "unknown_env_name"
    else:
        _env_name_part = env_name.replace("/", "_")

    log_dir = pathlib.Path(log_root) / _env_name_part / util.make_unique_timestamp()
    del _env_name_part


# TODO(shwang): Move these redundant configs into a `auto.env` Ingredient,
# similar to what the ILR project does.
@train_dagger_ex.named_config
def mountain_car():
    env_name = "MountainCar-v0"
    rollout_hint = "mountain_car"
    l2_weight = 0
    total_timesteps = 20000


@train_dagger_ex.named_config
def seals_mountain_car():
    env_name = "seals/MountainCar-v0"
    rollout_hint = "seals_mountain_car"
    l2_weight = 0
    total_timesteps = 20000


@train_dagger_ex.named_config
def cartpole():
    env_name = "CartPole-v1"
    rollout_hint = "cartpole"
    total_timesteps = 20000


@train_dagger_ex.named_config
def seals_cartpole():
    env_name = "seals/CartPole-v0"
    rollout_hint = "seals_cartpole"
    total_timesteps = 20000


@train_dagger_ex.named_config
def pendulum():
    env_name = "Pendulum-v0"
    rollout_hint = "pendulum"


@train_dagger_ex.named_config
def ant():
    env_name = "Ant-v2"
    rollout_hint = "ant"


@train_dagger_ex.named_config
def half_cheetah():
    env_name = "HalfCheetah-v2"
    rollout_hint = "half_cheetah"
    l2_weight = 0
    total_timesteps = 60000


@train_dagger_ex.named_config
def humanoid():
    env_name = "Humanoid-v2"
    rollout_hint = "humanoid"


@train_dagger_ex.named_config
def fast():
    total_timesteps = 50
    bc_train_kwargs = dict(
        n_batches=50,
    )
    n_episodes_eval = 1
    n_expert_demos = 1
