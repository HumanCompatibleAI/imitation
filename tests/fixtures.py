import os

import gym
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.util.networks import RunningNorm


CARTPOLE_ENV_NAME = "CartPole-v1"


@pytest.fixture(params=[1, 4])
def cartpole_venv(request) -> gym.Env:
    num_envs = request.param
    return DummyVecEnv([lambda: gym.make(CARTPOLE_ENV_NAME) for _ in range(num_envs)])


def train_cartpole_expert_policy(cartpole_venv) -> PPO:
    policy_kwargs = dict(features_extractor_class=NormalizeFeaturesExtractor,
                         features_extractor_kwargs=dict(normalize_class=RunningNorm))
    policy = PPO(policy=FeedForward32Policy, policy_kwargs=policy_kwargs,
                 env=VecNormalize(cartpole_venv, norm_obs=False), seed=0, batch_size=64, ent_coef=0.0,
                 learning_rate=0.0003, n_epochs=10, n_steps=64 // cartpole_venv.num_envs)
    policy.learn(100000)
    return policy


@pytest.fixture
def cached_cartpole_expert_policy(cartpole_venv, pytestconfig) -> PPO:
    cached_expert_path = str(pytestconfig.cache.makedir("experts") / CARTPOLE_ENV_NAME / "model.zip")
    try:
        expert_policy = PPO.load(cached_expert_path, cartpole_venv)
        return expert_policy
    except Exception as e:
        expert_policy = train_cartpole_expert_policy(cartpole_venv)
        expert_policy.save(cached_expert_path)
        return expert_policy
