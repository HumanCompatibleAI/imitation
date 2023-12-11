"""Fixtures common across algorithm tests."""
from typing import Sequence

import gymnasium as gym
import pytest
from stable_baselines3.common import envs
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from imitation.algorithms import bc
from imitation.data.types import TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies import serialize
from imitation.testing.expert_trajectories import (
    lazy_generate_expert_trajectories,
    make_expert_transition_loader,
)
from imitation.util import util

CARTPOLE_ENV_NAME = "seals/CartPole-v0"


@pytest.fixture
def cartpole_expert_policy(cartpole_venv: VecEnv) -> BasePolicy:
    return serialize.load_policy(
        "ppo-huggingface",
        cartpole_venv,
        env_name=CARTPOLE_ENV_NAME,
    )


@pytest.fixture
def cartpole_expert_trajectories(
    cartpole_expert_policy,
    cartpole_venv,
    pytestconfig,
    rng,
) -> Sequence[TrajectoryWithRew]:
    return lazy_generate_expert_trajectories(
        pytestconfig.cache.makedir("experts"),
        CARTPOLE_ENV_NAME,
        60,
        rng,
    )


PENDULUM_ENV_NAME = "Pendulum-v1"


@pytest.fixture
def cartpole_bc_trainer(
    pytestconfig,
    cartpole_venv,
    cartpole_expert_trajectories,
    rng,
):
    return bc.BC(
        observation_space=cartpole_venv.observation_space,
        action_space=cartpole_venv.action_space,
        batch_size=50,
        demonstrations=make_expert_transition_loader(
            cache_dir=pytestconfig.cache.makedir("experts"),
            batch_size=50,
            expert_data_type="transitions",
            env_name="seals/CartPole-v0",
            rng=rng,
            num_trajectories=60,
        ),
        custom_logger=None,
        rng=rng,
    )


@pytest.fixture
def pendulum_expert_trajectories(
    pytestconfig,
    rng,
) -> Sequence[TrajectoryWithRew]:
    return lazy_generate_expert_trajectories(
        pytestconfig.cache.makedir("experts"),
        PENDULUM_ENV_NAME,
        60,
        rng,
    )


@pytest.fixture
def pendulum_expert_policy(pendulum_venv) -> BasePolicy:
    return serialize.load_policy(
        "ppo-huggingface",
        pendulum_venv,
        env_name=PENDULUM_ENV_NAME,
    )


@pytest.fixture
def pendulum_venv(rng) -> VecEnv:
    return util.make_vec_env(
        PENDULUM_ENV_NAME,
        n_envs=8,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        rng=rng,
    )


@pytest.fixture
def pendulum_single_venv(rng) -> VecEnv:
    return util.make_vec_env(
        PENDULUM_ENV_NAME,
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        rng=rng,
    )


@pytest.fixture
def multi_obs_venv() -> VecEnv:
    def make_env():
        env = envs.SimpleMultiObsEnv(channel_last=False)
        return RolloutInfoWrapper(env)

    return DummyVecEnv([make_env, make_env])
