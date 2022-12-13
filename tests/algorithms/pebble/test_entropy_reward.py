"""Tests for `imitation.algorithms.entropy_reward`."""
import pickle
from unittest.mock import Mock

import numpy as np
import pytest
import torch as th
from gym.spaces import Box
from gym.spaces.space import Space

from imitation.algorithms.pebble.entropy_reward import (
    EntropyRewardNet,
    InsufficientObservations,
    PebbleStateEntropyReward,
)
from imitation.policies.replay_buffer_wrapper import (
    ReplayBufferAwareRewardFn,
    ReplayBufferView,
)
from imitation.util import util

SPACE = Box(-1, 1, shape=(1,))
PLACEHOLDER = np.empty(SPACE.shape)

BUFFER_SIZE = 20
K = 4
BATCH_SIZE = 8
VENVS = 2


def test_pebble_entropy_reward_returns_entropy_for_pretraining():
    expected_result = th.rand(BATCH_SIZE)
    observations = th.rand((BATCH_SIZE,) + SPACE.shape)
    entropy_fn = Mock()
    entropy_fn.return_value = expected_result
    learned_fn = Mock()

    reward_fn = PebbleStateEntropyReward(entropy_fn, learned_fn)
    reward = reward_fn(observations, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)

    np.testing.assert_allclose(reward, expected_result)
    entropy_fn.assert_called_once_with(
        observations,
        PLACEHOLDER,
        PLACEHOLDER,
        PLACEHOLDER,
    )


def test_pebble_entropy_reward_returns_learned_rew_on_insufficient_observations(rng):
    expected_result = th.rand(BATCH_SIZE)
    observations = th.rand((BATCH_SIZE,) + SPACE.shape)
    entropy_fn = Mock()
    entropy_fn.side_effect = InsufficientObservations("test error")
    learned_fn = Mock()
    learned_fn.return_value = expected_result

    reward_fn = PebbleStateEntropyReward(entropy_fn, learned_fn)
    reward = reward_fn(observations, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)

    np.testing.assert_allclose(reward, expected_result)
    learned_fn.assert_called_once_with(
        observations,
        PLACEHOLDER,
        PLACEHOLDER,
        PLACEHOLDER,
    )


def test_pebble_entropy_reward_function_returns_learned_reward_after_pre_training():
    expected_result = th.rand(BATCH_SIZE)
    observations = th.rand((BATCH_SIZE,) + SPACE.shape)
    entropy_fn = Mock()
    learned_fn = Mock()
    learned_fn.return_value = expected_result

    reward_fn = PebbleStateEntropyReward(entropy_fn, learned_fn)
    reward_fn.unsupervised_exploration_finish()
    reward = reward_fn(observations, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)

    np.testing.assert_allclose(reward, expected_result)
    learned_fn.assert_called_once_with(
        observations,
        PLACEHOLDER,
        PLACEHOLDER,
        PLACEHOLDER,
    )


def test_pebble_entropy_reward_propagates_on_replay_buffer_initialized():
    replay_buffer = replay_buffer_mock(np.empty((BUFFER_SIZE, VENVS) + SPACE.shape))
    entropy_fn = Mock(spec=ReplayBufferAwareRewardFn)
    learned_fn = Mock()

    reward_fn = PebbleStateEntropyReward(entropy_fn, learned_fn)
    reward_fn.on_replay_buffer_initialized(replay_buffer)

    entropy_fn.on_replay_buffer_initialized.assert_called_once_with(replay_buffer)


def test_entropy_reward_net_returns_entropy_for_pretraining(rng):
    observations = th.rand((BATCH_SIZE, *SPACE.shape))
    all_observations = rng.random((BUFFER_SIZE, VENVS) + SPACE.shape)
    reward_net = EntropyRewardNet(K, SPACE, SPACE)
    reward_net.on_replay_buffer_initialized(replay_buffer_mock(all_observations))

    # Act
    reward = reward_net.predict_processed(
        observations,
        PLACEHOLDER,
        PLACEHOLDER,
        PLACEHOLDER,
    )

    # Assert
    expected = util.compute_state_entropy(
        observations,
        all_observations.reshape(-1, *SPACE.shape),
        K,
    )
    np.testing.assert_allclose(reward, expected, rtol=0.005, atol=0.005)


def test_entropy_reward_net_raises_on_insufficient_observations(rng):
    observations = th.rand((BATCH_SIZE, *SPACE.shape))
    all_observations = rng.random((K - 1, 1) + SPACE.shape)
    reward_net = EntropyRewardNet(K, SPACE, SPACE)
    reward_net.on_replay_buffer_initialized(replay_buffer_mock(all_observations))

    # Act
    with pytest.raises(InsufficientObservations):
        reward_net.predict_processed(
            observations,
            PLACEHOLDER,
            PLACEHOLDER,
            PLACEHOLDER,
        )


def test_entropy_reward_net_can_pickle(rng):
    all_observations = np.empty((BUFFER_SIZE, VENVS, *SPACE.shape))
    replay_buffer = replay_buffer_mock(all_observations)
    reward_net = EntropyRewardNet(K, SPACE, SPACE)
    reward_net.on_replay_buffer_initialized(replay_buffer)

    # Act
    pickled = pickle.dumps(reward_net)
    reward_fn_deserialized = pickle.loads(pickled)
    reward_fn_deserialized.on_replay_buffer_initialized(replay_buffer)

    # Assert
    obs = th.rand(VENVS, *SPACE.shape)
    expected_result = reward_net(obs, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)
    actual_result = reward_fn_deserialized(obs, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)
    np.testing.assert_allclose(actual_result, expected_result)


def replay_buffer_mock(all_observations: np.ndarray, obs_space: Space = SPACE) -> Mock:
    buffer_view = ReplayBufferView(all_observations, lambda: slice(None))
    mock = Mock()
    mock.buffer_view = buffer_view
    mock.observation_space = obs_space
    mock.action_space = SPACE
    return mock
