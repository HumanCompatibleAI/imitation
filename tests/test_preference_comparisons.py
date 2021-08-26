"""Tests for the preference comparisons reward learning implementation."""

import pytest
from stable_baselines3 import PPO

from imitation.algorithms import preference_comparisons
from imitation.data import fragments, types
from imitation.policies import trainer
from imitation.rewards import reward_nets
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
    reward_net = reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)
    # verbose=1 suppresses SB3 logger configuration,
    # which conflicts with imitation logging
    agent = PPO("MlpPolicy", venv, verbose=1)
    agent_trainer = trainer.AgentTrainer(agent, reward_net)
    fragmenter = fragments.RandomFragmenter(
        fragment_length=2, num_pairs=2, seed=0, warning_threshold=0
    )
    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer, reward_net, timesteps=10, fragmenter=fragmenter
    )
    main_trainer.train(2)
