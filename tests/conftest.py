"""Fixtures common across tests."""

import pytest
import torch
import pickle
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.util import logger
from imitation.util.networks import RunningNorm


CARTPOLE_ENV_NAME = "CartPole-v1"


@pytest.fixture(params=[1, 4])
def cartpole_venv(request) -> gym.Env:
    num_envs = request.param
    return DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(CARTPOLE_ENV_NAME)) for _ in range(num_envs)])


def train_cartpole_expert_policy(cartpole_venv) -> PPO:
    policy_kwargs = dict(features_extractor_class=NormalizeFeaturesExtractor,
                         features_extractor_kwargs=dict(normalize_class=RunningNorm))
    policy = PPO(policy=FeedForward32Policy, policy_kwargs=policy_kwargs,
                 env=VecNormalize(cartpole_venv, norm_obs=False), seed=0, batch_size=64, ent_coef=0.0,
                 learning_rate=0.0003, n_epochs=10, n_steps=64 // cartpole_venv.num_envs)
    policy.learn(100000)
    return policy


@pytest.fixture
def cartpole_expert_policy(cartpole_venv, pytestconfig) -> PPO:
    cached_expert_path = str(pytestconfig.cache.makedir("experts") / CARTPOLE_ENV_NAME / "model.zip")
    try:
        expert_policy = PPO.load(cached_expert_path, cartpole_venv)
        return expert_policy.policy
    except Exception as e:
        expert_policy = train_cartpole_expert_policy(cartpole_venv)
        expert_policy.save(cached_expert_path)
        return expert_policy.policy


@pytest.fixture
def cartpole_expert_trajectories(cartpole_expert_policy, cartpole_venv, pytestconfig):
    rollouts_path = str(pytestconfig.cache.makedir("experts") / CARTPOLE_ENV_NAME / "rollout.pkl")
    try:
        with open(rollouts_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        rollout.rollout_and_save(rollouts_path, cartpole_expert_policy, cartpole_venv,
                                 rollout.make_sample_until(min_timesteps=2000, min_episodes=57))
        with open(rollouts_path, "rb") as f:  # TODO: not re-loading the trajectory would be nicer here
            return pickle.load(f)


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
