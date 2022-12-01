import pickle
from unittest.mock import patch

import numpy as np
import torch as th
from gym.spaces import Discrete
from stable_baselines3.common.preprocessing import get_obs_shape

from imitation.algorithms.pebble.entropy_reward import PebbleStateEntropyReward
from imitation.policies.replay_buffer_wrapper import ReplayBufferView
from imitation.util import util

SPACE = Discrete(4)
PLACEHOLDER = np.empty(get_obs_shape(SPACE))

BUFFER_SIZE = 20
K = 4
BATCH_SIZE = 8
VENVS = 2


def test_state_entropy_reward_returns_entropy(rng):
    obs_shape = get_obs_shape(SPACE)
    all_observations = rng.random((BUFFER_SIZE, VENVS, *obs_shape))


    reward_fn = PebbleStateEntropyReward(K, SPACE)
    reward_fn.set_replay_buffer(ReplayBufferView(all_observations, lambda: slice(None)), obs_shape)

    # Act
    observations = rng.random((BATCH_SIZE, *obs_shape))
    reward = reward_fn(observations, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)

    # Assert
    expected = util.compute_state_entropy(
        observations, all_observations.reshape(-1, *obs_shape), K
    )
    expected_normalized = reward_fn.entropy_stats.normalize(
        th.as_tensor(expected)
    ).numpy()
    np.testing.assert_allclose(reward, expected_normalized)


def test_state_entropy_reward_returns_normalized_values():
    with patch("imitation.util.util.compute_state_entropy") as m:
        # mock entropy computation so that we can test only stats collection in this test
        m.side_effect = lambda obs, all_obs, k: obs

        reward_fn = PebbleStateEntropyReward(K, SPACE)
        all_observations = np.empty((BUFFER_SIZE, VENVS, *get_obs_shape(SPACE)))
        reward_fn.set_replay_buffer(
            ReplayBufferView(all_observations, lambda: slice(None)),
            get_obs_shape(SPACE)
        )

        dim = 8
        shift = 3
        scale = 2

        # Act
        for _ in range(1000):
            state = th.randn(dim) * scale + shift
            reward_fn(state, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)

        normalized_reward = reward_fn(
            np.zeros(dim), PLACEHOLDER, PLACEHOLDER, PLACEHOLDER
        )

        # Assert
        np.testing.assert_allclose(
            normalized_reward,
            np.repeat(-shift / scale, dim),
            rtol=0.05,
            atol=0.05,
        )


def test_state_entropy_reward_can_pickle():
    all_observations = np.empty((BUFFER_SIZE, VENVS, *get_obs_shape(SPACE)))
    replay_buffer = ReplayBufferView(all_observations, lambda: slice(None))

    obs1 = np.random.rand(VENVS, *get_obs_shape(SPACE))
    reward_fn = PebbleStateEntropyReward(K, SPACE)
    reward_fn.set_replay_buffer(replay_buffer, get_obs_shape(SPACE))
    reward_fn(obs1, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)

    # Act
    pickled = pickle.dumps(reward_fn)
    reward_fn_deserialized = pickle.loads(pickled)
    reward_fn_deserialized.set_replay_buffer(replay_buffer)

    # Assert
    obs2 = np.random.rand(VENVS, *get_obs_shape(SPACE))
    expected_result = reward_fn(obs2, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)
    actual_result = reward_fn_deserialized(obs2, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)
    np.testing.assert_allclose(actual_result, expected_result)
