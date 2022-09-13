"""Fixtures common across tests."""
import os
import pickle
import traceback
import warnings
from typing import Sequence

import gym
import numpy as np
import pytest
import torch
from filelock import FileLock
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from imitation.data import rollout, types
from imitation.data.types import TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger

CARTPOLE_ENV_NAME = "CartPole-v1"


def load_or_rollout_trajectories(
    cache_path,
    policy,
    venv,
    rng,
) -> Sequence[TrajectoryWithRew]:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with FileLock(cache_path + ".lock"):
        try:
            return types.load_with_rewards(cache_path)
        except (OSError, pickle.PickleError):  # pragma: no cover
            warnings.warn(
                "Recomputing expert trajectories due to the following error when "
                "trying to load them:\n" + traceback.format_exc(),
            )
            rollouts = rollout.rollout(
                policy,
                venv,
                rollout.make_sample_until(min_timesteps=2000, min_episodes=57),
                rng=rng,
            )
            types.save(cache_path, rollouts)
            return rollouts


@pytest.fixture(params=[1, 4])
def cartpole_venv(request) -> VecEnv:
    num_envs = request.param
    return DummyVecEnv(
        [
            lambda: RolloutInfoWrapper(gym.make(CARTPOLE_ENV_NAME))
            for _ in range(num_envs)
        ],
    )


@pytest.fixture
def cartpole_expert_policy():
    return PPO.load(
        load_from_hub(
            "HumanCompatibleAI/ppo-seals-CartPole-v0",
            "ppo-seals-CartPole-v0.zip",
        ),
    ).policy


@pytest.fixture
def cartpole_expert_trajectories(
    cartpole_expert_policy,
    cartpole_venv,
    pytestconfig,
    rng,
) -> Sequence[TrajectoryWithRew]:
    rollouts_path = str(
        pytestconfig.cache.makedir("experts") / CARTPOLE_ENV_NAME / "rollout.npz",
    )
    return load_or_rollout_trajectories(
        rollouts_path,
        cartpole_expert_policy,
        cartpole_venv,
        rng,
    )


PENDULUM_ENV_NAME = "Pendulum-v1"


@pytest.fixture
def pendulum_venv() -> VecEnv:
    return DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(PENDULUM_ENV_NAME))] * 8)


@pytest.fixture
def pendulum_expert_policy() -> BasePolicy:
    return PPO.load(
        load_from_hub(
            "HumanCompatibleAI/ppo-Pendulum-v1",
            "ppo-Pendulum-v1.zip",
        ),
    ).policy


@pytest.fixture
def pendulum_expert_trajectories(
    pendulum_expert_policy,
    pendulum_venv,
    pytestconfig,
    rng,
) -> Sequence[TrajectoryWithRew]:
    rollouts_path = str(
        pytestconfig.cache.makedir("experts") / PENDULUM_ENV_NAME / "rollout.npz",
    )
    return load_or_rollout_trajectories(
        rollouts_path,
        pendulum_expert_policy,
        pendulum_venv,
        rng=rng,
    )


@pytest.fixture(scope="session", autouse=True)
def torch_single_threaded():
    """Make PyTorch execute code single-threaded.

    This allows us to run the test suite with greater across-test parallelism.
    This is faster, since:
        - There are diminishing returns to more threads within a test.
        - Many tests cannot be multi-threaded (e.g. most not using PyTorch training),
          and we have to set between-test parallelism based on peak resource
          consumption of tests to avoid spurious failures.
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


@pytest.fixture()
def custom_logger(tmpdir: str) -> logger.HierarchicalLogger:
    return logger.configure(tmpdir)


@pytest.fixture()
def rng_fixed() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng()
