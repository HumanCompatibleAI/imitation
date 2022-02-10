"""Fixtures common across tests."""
import pickle
from typing import Callable, List

import gym
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.util import logger
from imitation.util.networks import RunningNorm

CARTPOLE_ENV_NAME = "CartPole-v1"


def load_or_train_ppo(
    cache_path: str,
    training_function: Callable[[gym.Env], PPO],
    venv,
) -> PPO:
    try:
        return PPO.load(cache_path, venv)
    except OSError:
        pass  # File not found, or path is a directory
    except AssertionError:
        pass  # Model was stored with an older version of stable baselines
    except pickle.PickleError:
        pass  # File contains something, that can not be unpickled
    expert = training_function(venv)
    expert.save(cache_path)
    return expert


def load_or_rollout_trajectories(cache_path, policy, venv) -> List[TrajectoryWithRew]:
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except OSError:
        pass  # File not found, or path is a directory
    except AssertionError:
        pass  # Model was stored with an older version of stable baselines
    except pickle.PickleError:
        pass  # File contains something, that can not be unpickled
    rollout.rollout_and_save(
        cache_path,
        policy,
        venv,
        rollout.make_sample_until(min_timesteps=2000, min_episodes=57),
    )
    with open(cache_path, "rb") as f:
        return pickle.load(f)  # TODO: not re-loading the trajectory would be nicer here


@pytest.fixture(params=[1, 4])
def cartpole_venv(request) -> gym.Env:
    num_envs = request.param
    return DummyVecEnv(
        [
            lambda: RolloutInfoWrapper(gym.make(CARTPOLE_ENV_NAME))
            for _ in range(num_envs)
        ],
    )


def train_cartpole_expert(cartpole_venv) -> PPO:
    policy_kwargs = dict(
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    )
    policy = PPO(
        policy=FeedForward32Policy,
        policy_kwargs=policy_kwargs,
        env=VecNormalize(cartpole_venv, norm_obs=False),
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64 // cartpole_venv.num_envs,
    )
    policy.learn(100000)
    return policy


@pytest.fixture
def cartpole_expert_policy(cartpole_venv, pytestconfig) -> BasePolicy:
    cached_expert_path = str(
        pytestconfig.cache.makedir("experts") / CARTPOLE_ENV_NAME / "model.zip",
    )
    return load_or_train_ppo(
        cached_expert_path,
        train_cartpole_expert,
        cartpole_venv,
    ).policy


@pytest.fixture
def cartpole_expert_trajectories(
    cartpole_expert_policy,
    cartpole_venv,
    pytestconfig,
) -> List[TrajectoryWithRew]:
    rollouts_path = str(
        pytestconfig.cache.makedir("experts") / CARTPOLE_ENV_NAME / "rollout.pkl",
    )
    return load_or_rollout_trajectories(
        rollouts_path,
        cartpole_expert_policy,
        cartpole_venv,
    )


PENDULUM_ENV_NAME = "Pendulum-v1"


@pytest.fixture
def pendulum_venv() -> gym.Env:
    return DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(PENDULUM_ENV_NAME))] * 8)


def train_pendulum_expert(pendulum_venv) -> PPO:
    policy_kwargs = dict(
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    )
    policy = PPO(
        policy=FeedForward32Policy,
        policy_kwargs=policy_kwargs,
        env=VecNormalize(pendulum_venv, norm_obs=False),
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0001,
        n_epochs=10,
        n_steps=4096 // pendulum_venv.num_envs,
        gamma=0.9,
    )
    policy.learn(200000)
    return policy


@pytest.fixture
def pendulum_expert_policy(pendulum_venv, pytestconfig) -> BasePolicy:
    cached_expert_path = str(
        pytestconfig.cache.makedir("experts") / PENDULUM_ENV_NAME / "model.zip",
    )
    return load_or_train_ppo(
        cached_expert_path,
        train_pendulum_expert,
        pendulum_venv,
    ).policy


@pytest.fixture
def pendulum_expert_trajectories(
    pendulum_expert_policy,
    pendulum_venv,
    pytestconfig,
) -> List[TrajectoryWithRew]:
    rollouts_path = str(
        pytestconfig.cache.makedir("experts") / PENDULUM_ENV_NAME / "rollout.pkl",
    )
    return load_or_rollout_trajectories(
        rollouts_path,
        pendulum_expert_policy,
        pendulum_venv,
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
