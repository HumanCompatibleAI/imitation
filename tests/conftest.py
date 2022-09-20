"""Fixtures common across tests."""
import os
import pickle
import traceback
import warnings
from typing import Sequence

import gym
import huggingface_sb3 as hfsb3
import pytest
import seals  # noqa: F401
import torch
from filelock import FileLock
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from imitation.data import rollout, types, wrappers
from imitation.data.types import TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies import serialize
from imitation.util import logger, util

CARTPOLE_ENV_NAME = "seals/CartPole-v0"


def get_expert_trajectories(pytestconfig, env_name: str, min_episodes: int = 60):
    trajectories_path = str(
        pytestconfig.cache.makedir("experts")
        / hfsb3.EnvironmentName(env_name)
        / "rollout.npz",
    )

    os.makedirs(os.path.dirname(trajectories_path), exist_ok=True)
    with FileLock(trajectories_path + ".lock"):
        try:
            trajectories = types.load_with_rewards(trajectories_path)
        except (FileNotFoundError, pickle.PickleError):  # pragma: no cover
            warnings.warn(
                "Recomputing expert trajectories due to the following error when "
                "trying to load them:\n" + traceback.format_exc(),
            )

            env = util.make_vec_env(
                env_name,
                log_dir=None,
                post_wrappers=[lambda env, i: wrappers.RolloutInfoWrapper(env)],
            )
            trajectories = rollout.rollout(
                serialize.load_policy("ppo-huggingface", env, env_name=env_name),
                env,
                rollout.make_sample_until(min_episodes=min_episodes),
            )
            env.close()
            types.save(trajectories_path, trajectories)
        return trajectories


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
    return get_expert_trajectories(pytestconfig, CARTPOLE_ENV_NAME)


PENDULUM_ENV_NAME = "Pendulum-v1"


@pytest.fixture
def pendulum_venv() -> VecEnv:
    return DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(PENDULUM_ENV_NAME))] * 8)


@pytest.fixture
def pendulum_expert_policy(pendulum_venv) -> BasePolicy:
    return serialize.load_policy(
        "ppo-huggingface",
        pendulum_venv,
        env_name=PENDULUM_ENV_NAME,
    )


@pytest.fixture
def pendulum_expert_trajectories(
    pytestconfig,
) -> Sequence[TrajectoryWithRew]:
    return get_expert_trajectories(pytestconfig, PENDULUM_ENV_NAME)


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
