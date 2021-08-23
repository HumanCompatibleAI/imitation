"""Tests for the preference comparisons reward learning implementation."""

import pytest
from stable_baselines3 import PPO

from imitation.algorithms.preference_comparisons import PreferenceComparisons
from imitation.data import types
from imitation.data.fragments import RandomFragmenter
from imitation.policies.trainer import AgentTrainer
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util import util


@pytest.fixture
def expert_trajectories():
    trajs = types.load("tests/data/expert_models/cartpole_0/rollouts/final.pkl")
    return trajs


def test_trainer_no_crash():
    venv = util.make_vec_env(
        "CartPole-v1",
        n_envs=1,
    )
    reward_net = BasicRewardNet(venv.observation_space, venv.action_space)
    # verbose=1 suppresses SB3 logger configuration,
    # which conflicts with imitation logging
    agent = PPO("MlpPolicy", venv, verbose=1)
    agent_trainer = AgentTrainer(agent, reward_net)
    fragmenter = RandomFragmenter(fragment_length=2, num_pairs=2, seed=0)
    trainer = PreferenceComparisons(agent_trainer, reward_net, 10, fragmenter)
    trainer.train(2)
