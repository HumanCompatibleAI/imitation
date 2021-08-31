"""Tests for the preference comparisons reward learning implementation."""

import numpy as np
import pytest
import stable_baselines3

from imitation.algorithms import preference_comparisons
from imitation.policies import trainer
from imitation.rewards import reward_nets
from imitation.util import util


@pytest.fixture
def venv():
    return util.make_vec_env(
        "CartPole-v1",
        n_envs=1,
    )


@pytest.fixture
def reward_net(venv):
    return reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)


@pytest.fixture
def agent(venv):
    # verbose=1 suppresses SB3 logger configuration,
    # which conflicts with imitation logging
    return stable_baselines3.PPO(
        "MlpPolicy", venv, verbose=1, n_epochs=1, batch_size=2, n_steps=10
    )


@pytest.fixture
def fragmenter():
    return preference_comparisons.RandomFragmenter(
        fragment_length=2, num_pairs=2, seed=0, warning_threshold=0
    )


@pytest.fixture
def agent_trainer(agent, reward_net):
    return trainer.AgentTrainer(agent, reward_net)


def test_trainer_no_crash(agent_trainer, reward_net, fragmenter):
    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer, reward_net, sample_steps=10, fragmenter=fragmenter
    )
    main_trainer.train(2)


def test_discount_rate_no_crash(agent_trainer, reward_net, fragmenter):
    # also use a non-zero noise probability to ensure that doesn't cause errors
    reward_trainer = preference_comparisons.CrossEntropyRewardTrainer(
        reward_net, noise_prob=0.1, discount_factor=0.9
    )
    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        reward_net,
        sample_steps=10,
        fragmenter=fragmenter,
        reward_trainer=reward_trainer,
    )
    main_trainer.train(2)


def test_synthetic_gatherer_deterministic(agent_trainer, fragmenter):
    gatherer = preference_comparisons.SyntheticGatherer(temperature=0)
    trajectories = agent_trainer.sample(10)
    fragments = fragmenter(trajectories)
    preferences1 = gatherer(fragments)
    preferences2 = gatherer(fragments)
    assert np.all(preferences1 == preferences2)
