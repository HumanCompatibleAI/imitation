"""Fixtures common across algorithm tests."""
from typing import Sequence

import pytest
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

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
) -> Sequence[TrajectoryWithRew]:
    return lazy_generate_expert_trajectories(
        pytestconfig.cache.makedir("experts"),
        CARTPOLE_ENV_NAME,
        60,
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
            pytestconfig.cache.makedir("experts"),
            50,
            "transitions",
            "seals/CartPole-v0",
        ),
        custom_logger=None,
        rng=rng,
    )


@pytest.fixture
def pendulum_expert_trajectories(
    pytestconfig,
) -> Sequence[TrajectoryWithRew]:
    return lazy_generate_expert_trajectories(
        pytestconfig.cache.makedir("experts"),
        PENDULUM_ENV_NAME,
        60,
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
