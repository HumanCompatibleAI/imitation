"""Fixtures common across tests."""
import os
import pickle
import traceback
import warnings
from typing import Callable, Optional, Sequence

import gym
import pytest
import torch
from filelock import FileLock
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize
from stable_baselines3.ppo import MlpPolicy

from imitation.data import rollout, types
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
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with FileLock(cache_path + ".lock"):
        try:
            return PPO.load(cache_path, venv)
        except (OSError, AssertionError, pickle.PickleError):  # pragma: no cover
            # Note, when loading models from older stable-baselines versions, we can get
            # AssertionErrors.
            warnings.warn(
                "Retraining expert policy due to the following error when trying"
                " to load it:\n" + traceback.format_exc(),
            )
            expert = training_function(venv)
            if expert is None:
                pytest.fail("Failed to train expert!")
            expert.save(cache_path)
            return expert


def load_or_rollout_trajectories(
    cache_path,
    policy,
    venv,
) -> Sequence[TrajectoryWithRew]:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with FileLock(cache_path + ".lock"):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except (OSError, pickle.PickleError):  # pragma: no cover
            warnings.warn(
                "Recomputing expert trajectories due to the following error when "
                "trying to load them:\n" + traceback.format_exc(),
            )
            rollouts = rollout.rollout(
                policy,
                venv,
                rollout.make_sample_until(min_timesteps=2000, min_episodes=57),
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


def train_cartpole_expert(cartpole_env) -> Optional[PPO]:  # pragma: no cover
    """Trains an expert on a cartpole environment.

    Args:
        cartpole_env: The cartpole environment to use for training. Will only work with
            CartPole-v1

    Returns:
        The trained cartpole expert or None if training failed even after 10 retries.
    """
    policy_kwargs = dict(
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    )
    for attempt_nr in range(10):
        policy = PPO(
            policy=FeedForward32Policy,
            policy_kwargs=policy_kwargs,
            env=VecNormalize(cartpole_env, norm_obs=False),
            seed=attempt_nr,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=10,
            n_steps=64 // cartpole_env.num_envs,
        )
        policy.learn(100000)
        mean_reward, _ = evaluate_policy(policy, cartpole_env, 10)
        if mean_reward >= 500:
            return policy
    return None


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
) -> Sequence[TrajectoryWithRew]:
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
def pendulum_venv() -> VecEnv:
    return DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(PENDULUM_ENV_NAME))] * 8)


def train_pendulum_expert(pendulum_env) -> Optional[PPO]:  # pragma: no cover
    for attempt_nr in range(10):
        policy = PPO(
            policy=MlpPolicy,
            env=VecNormalize(pendulum_env, norm_obs=False),
            seed=attempt_nr,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=1e-3,
            n_epochs=10,
            n_steps=1024,
            gamma=0.9,
            gae_lambda=0.95,
            use_sde=True,
            sde_sample_freq=4,
        )
        policy.learn(int(1e5))
        mean_reward, _ = evaluate_policy(policy, pendulum_env, 10)
        if mean_reward >= -185:
            return policy
    return None


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
) -> Sequence[TrajectoryWithRew]:
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
