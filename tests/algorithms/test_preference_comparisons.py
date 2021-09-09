"""Tests for the preference comparisons reward learning implementation."""

import re

import numpy as np
import pytest
import seals  # noqa: F401
import stable_baselines3

from imitation.algorithms import preference_comparisons
from imitation.data import types
from imitation.rewards import reward_nets
from imitation.util import util


@pytest.fixture
def venv():
    return util.make_vec_env(
        "seals/CartPole-v0",
        n_envs=1,
    )


@pytest.fixture
def reward_net(venv):
    return reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)


@pytest.fixture
def agent(venv):
    return stable_baselines3.PPO(
        "MlpPolicy", venv, n_epochs=1, batch_size=2, n_steps=10
    )


@pytest.fixture
def fragmenter():
    return preference_comparisons.RandomFragmenter(seed=0, warning_threshold=0)


@pytest.fixture
def agent_trainer(agent, reward_net):
    return preference_comparisons.AgentTrainer(agent, reward_net)


def test_missing_environment(agent):
    # Create an agent that doesn't have its environment set.
    # More realistically, this can happen when loading a stored agent.
    agent.env = None
    with pytest.raises(
        ValueError, match="The environment for the agent algorithm must be set."
    ):
        preference_comparisons.AgentTrainer(agent, reward_net)


def test_transitions_left_in_buffer(agent_trainer):
    # Faster to just set the counter than to actually fill the buffer
    # with transitions.
    agent_trainer.buffering_wrapper.n_transitions = 2
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "There are 2 transitions left in the buffer. "
            "Call AgentTrainer.sample() first to clear them."
        ),
    ):
        agent_trainer.train(steps=1)


PREFERENCE_COMPARISON_CONFIGS = [
    {},
    {
        "preferences_path": "tests/testdata/preferences/cartpole/preferences.pkl",
    },
]


@pytest.mark.parametrize("config", PREFERENCE_COMPARISON_CONFIGS)
def test_trainer_no_crash(agent_trainer, reward_net, fragmenter, custom_logger, config):
    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        reward_net,
        transition_oversampling=2,
        fragment_length=2,
        comparisons_per_iteration=2,
        fragmenter=fragmenter,
        custom_logger=custom_logger,
        **config,
    )
    main_trainer.train(10, 3)


def test_discount_rate_no_crash(agent_trainer, reward_net, fragmenter, custom_logger):
    # also use a non-zero noise probability to check that doesn't cause errors
    reward_trainer = preference_comparisons.CrossEntropyRewardTrainer(
        reward_net, noise_prob=0.1, discount_factor=0.9
    )
    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        reward_net,
        transition_oversampling=2,
        fragment_length=2,
        comparisons_per_iteration=2,
        fragmenter=fragmenter,
        reward_trainer=reward_trainer,
        custom_logger=custom_logger,
    )
    main_trainer.train(10, 3)


def test_synthetic_gatherer_deterministic(agent_trainer, fragmenter):
    gatherer = preference_comparisons.SyntheticGatherer(temperature=0)
    trajectories = agent_trainer.sample(10)
    fragments = fragmenter(trajectories, fragment_length=2, num_pairs=2)
    preferences1 = gatherer(fragments)
    preferences2 = gatherer(fragments)
    assert np.all(preferences1 == preferences2)


def test_fragments_terminal(fragmenter):
    trajectories = [
        types.TrajectoryWithRew(
            obs=np.arange(4),
            acts=np.zeros((3,)),
            rews=np.zeros((3,)),
            infos=None,
            terminal=True,
        ),
        types.TrajectoryWithRew(
            obs=np.arange(3),
            acts=np.zeros((2,)),
            rews=np.zeros((2,)),
            infos=None,
            terminal=False,
        ),
    ]
    for _ in range(5):
        for frags in fragmenter(trajectories, fragment_length=2, num_pairs=2):
            for frag in frags:
                assert (frag.obs[-1] == 3) == frag.terminal


def test_fragments_too_short_error(agent_trainer):
    trajectories = agent_trainer.sample(2)
    fragmenter = preference_comparisons.RandomFragmenter(
        seed=0,
        warning_threshold=0,
    )
    with pytest.raises(
        ValueError,
        match="No trajectories are long enough for the desired fragment length.",
    ):
        # the only important bit is that fragment_length is higher than
        # we'll ever reach
        fragmenter(trajectories, fragment_length=10000, num_pairs=2)


def test_preference_dataset_errors(agent_trainer, fragmenter):
    dataset = preference_comparisons.PreferenceDataset()
    trajectories = agent_trainer.sample(2)
    fragments = fragmenter(trajectories, fragment_length=2, num_pairs=2)
    # just create something with a different shape:
    preferences = np.empty(len(fragments) + 1, dtype=np.float32)
    with pytest.raises(ValueError, match="Unexpected preferences shape"):
        dataset.push(fragments, preferences)

    # Now test dtype
    preferences = np.empty(len(fragments), dtype=np.float64)
    with pytest.raises(ValueError, match="preferences should have dtype float32"):
        dataset.push(fragments, preferences)
