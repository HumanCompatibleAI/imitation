"""Tests for `imitation.algorithms.entropy_reward`."""

import pickle
from unittest.mock import Mock, patch

import numpy as np
import torch as th
from gym.spaces import Discrete

from imitation.algorithms.pebble.entropy_reward import PebbleStateEntropyReward
from imitation.policies.replay_buffer_wrapper import ReplayBufferView
from imitation.util import util

SPACE = Discrete(4)
OBS_SHAPE = (1,)
PLACEHOLDER = np.empty(OBS_SHAPE)

BUFFER_SIZE = 20
K = 4
BATCH_SIZE = 8
VENVS = 2


def test_pebble_entropy_reward_returns_entropy_for_pretraining(rng):
    all_observations = rng.random((BUFFER_SIZE, VENVS, *OBS_SHAPE))

    reward_fn = PebbleStateEntropyReward(Mock(), K)
    reward_fn.on_replay_buffer_initialized(
        replay_buffer_mock(
            ReplayBufferView(all_observations, lambda: slice(None)),
            OBS_SHAPE,
        )
    )

    # Act
    observations = th.rand((BATCH_SIZE, *OBS_SHAPE))
    reward = reward_fn(observations, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)

    # Assert
    expected = util.compute_state_entropy(
        observations,
        all_observations.reshape(-1, *OBS_SHAPE),
        K,
    )
    expected_normalized = reward_fn.entropy_stats.normalize(
        th.as_tensor(expected),
    ).numpy()
    np.testing.assert_allclose(reward, expected_normalized)


def test_pebble_entropy_reward_returns_normalized_values_for_pretraining():
    with patch("imitation.util.util.compute_state_entropy") as m:
        # mock entropy computation so that we can test
        # only stats collection in this test
        m.side_effect = lambda obs, all_obs, k: obs

        reward_fn = PebbleStateEntropyReward(Mock(), K)
        all_observations = np.empty((BUFFER_SIZE, VENVS, *OBS_SHAPE))
        reward_fn.on_replay_buffer_initialized(
            replay_buffer_mock(
                ReplayBufferView(all_observations, lambda: slice(None)),
                OBS_SHAPE,
            )
        )

        dim = 8
        shift = 3
        scale = 2

        # Act
        for _ in range(1000):
            state = th.randn(dim) * scale + shift
            reward_fn(state, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)

        normalized_reward = reward_fn(
            np.zeros(dim),
            PLACEHOLDER,
            PLACEHOLDER,
            PLACEHOLDER,
        )

        # Assert
        np.testing.assert_allclose(
            normalized_reward,
            np.repeat(-shift / scale, dim),
            rtol=0.05,
            atol=0.05,
        )


def test_pebble_entropy_reward_function_returns_learned_reward_after_pre_training():
    expected_reward = np.ones(1)
    learned_reward_mock = Mock()
    learned_reward_mock.return_value = expected_reward
    reward_fn = PebbleStateEntropyReward(learned_reward_mock)
    # move all the way to the last state
    reward_fn.unsupervised_exploration_finish()

    # Act
    observations = np.ones((BATCH_SIZE, *OBS_SHAPE))
    reward = reward_fn(observations, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)

    # Assert
    assert reward == expected_reward
    learned_reward_mock.assert_called_once_with(
        observations,
        PLACEHOLDER,
        PLACEHOLDER,
        PLACEHOLDER,
    )


def test_pebble_entropy_reward_can_pickle():
    all_observations = np.empty((BUFFER_SIZE, VENVS, *OBS_SHAPE))
    replay_buffer = ReplayBufferView(all_observations, lambda: slice(None))

    obs1 = np.random.rand(VENVS, *OBS_SHAPE)
    reward_fn = PebbleStateEntropyReward(reward_fn_stub, K)
    reward_fn.on_replay_buffer_initialized(replay_buffer_mock(replay_buffer, OBS_SHAPE))
    reward_fn(obs1, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)

    # Act
    pickled = pickle.dumps(reward_fn)
    reward_fn_deserialized = pickle.loads(pickled)
    reward_fn_deserialized.on_replay_buffer_initialized(
        replay_buffer_mock(replay_buffer, OBS_SHAPE)
    )

    # Assert
    obs2 = np.random.rand(VENVS, *OBS_SHAPE)
    expected_result = reward_fn(obs2, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)
    actual_result = reward_fn_deserialized(obs2, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)
    np.testing.assert_allclose(actual_result, expected_result)


def reward_fn_stub(state, action, next_state, done):
    return state


def replay_buffer_mock(buffer_view: ReplayBufferView, obs_shape: tuple) -> Mock:
    replay_buffer_mock = Mock()
    replay_buffer_mock.buffer_view = buffer_view
    replay_buffer_mock.obs_shape = obs_shape
    return replay_buffer_mock
