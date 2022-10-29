"""Configuration settings for train_dagger, training DAgger from synthetic demos."""

import pathlib

import sacred
import torch as th

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
        rollout_round_min_episodes=None,  # use default value
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
    expert_dir = str(
        pathlib.Path.home() / "imitation/output/train_experts/"
        "2022-09-05T18:27:27-07:00/seals_ant_1/"
    )
    demonstrations = dict(
        rollout_path=expert_dir + "rollouts/final.pkl",
    )
    expert = {
        "policy_type": "ppo",
        "loader_kwargs": {
            "path": expert_dir + "policies/final/",
        },
    }
    del expert_dir


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
    expert_dir = str(
        pathlib.Path.home() / "imitation/output/train_experts/"
        "2022-09-05T18:27:27-07:00/seals_half_cheetah_1/"
    )
    demonstrations = dict(
        rollout_path=expert_dir + "rollouts/final.pkl",
    )
    expert = {
        "policy_type": "ppo",
        "loader_kwargs": {
            "path": expert_dir + "policies/final/",
        },
    }
    del expert_dir


@train_imitation_ex.named_config
def seals_hopper():
    common = dict(env_name="seals/Hopper-v0")
    expert_dir = str(
        pathlib.Path.home() / "imitation/output/train_experts/"
        "2022-10-11T06:27:42-07:00/seals_hopper_2/"
    )
    demonstrations = dict(
        rollout_path=expert_dir + "rollouts/final.pkl",
    )
    expert = {
        "policy_type": "ppo",
        "loader_kwargs": {
            "path": expert_dir + "policies/final/",
        },
    }
    del expert_dir


@train_imitation_ex.named_config
def seals_swimmer():
    common = dict(env_name="seals/Swimmer-v0")
    expert_dir = str(
        pathlib.Path.home() / "/imitation/output/train_experts/"
        "2022-10-11T06:27:42-07:00/seals_swimmer_4/"
    )
    demonstrations = dict(
        rollout_path=expert_dir + "rollouts/final.pkl",
    )
    expert = {
        "policy_type": "ppo",
        "loader_kwargs": {
            "path": expert_dir + "policies/final/",
        },
    }
    del expert_dir


@train_imitation_ex.named_config
def seals_walker():
    common = dict(env_name="seals/Walker2d-v0")
    expert_dir = str(
        pathlib.Path.home() / "imitation/output/train_experts/"
        "2022-10-11T06:27:42-07:00/seals_walker_8/"
    )
    demonstrations = dict(
        rollout_path=expert_dir + "rollouts/final.pkl",
    )
    expert = {
        "policy_type": "ppo",
        "loader_kwargs": {
            "path": expert_dir + "policies/final/",
        },
    }
    del expert_dir


@train_imitation_ex.named_config
def humanoid():
    common = dict(env_name="Humanoid-v2")


@train_imitation_ex.named_config
def seals_humanoid():
    common = dict(env_name="seals/Humanoid-v0")


@train_imitation_ex.named_config
def fast():
    dagger = dict(total_timesteps=50)
    bc_train_kwargs = dict(n_batches=50)
