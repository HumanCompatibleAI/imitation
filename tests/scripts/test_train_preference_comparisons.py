"""Tests train_preferences_comparisons helper methods."""

from unittest.mock import Mock, patch

import numpy as np
import torch as th
from gym import Space
from gym.spaces import Box

from imitation.policies.replay_buffer_wrapper import ReplayBufferView
from imitation.scripts.train_preference_comparisons import create_pebble_reward_fn

K = 4
SPACE = Box(-1, 1, shape=(1,))
BUFFER_SIZE = 20
VENVS = 2
PLACEHOLDER = np.empty(SPACE.shape)


def test_creates_normalized_entropy_pebble_reward():
    with patch("imitation.util.util.compute_state_entropy") as m:
        # mock entropy computation so that we can test
        # only stats collection in this test
        m.side_effect = lambda obs, all_obs, k: obs

        reward_fn = create_pebble_reward_fn(reward_fn_stub, K, SPACE, SPACE)

        all_observations = np.empty((BUFFER_SIZE, VENVS, *SPACE.shape))
        reward_fn.on_replay_buffer_initialized(replay_buffer_mock(all_observations))

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

        # Just to make coverage happy:
        reward_fn_stub(state, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)


def reward_fn_stub(state, action, next_state, done):
    return state


def replay_buffer_mock(all_observations: np.ndarray, obs_space: Space = SPACE) -> Mock:
    buffer_view = ReplayBufferView(all_observations, lambda: slice(None))
    mock = Mock()
    mock.buffer_view = buffer_view
    mock.observation_space = obs_space
    mock.action_space = SPACE
    return mock
