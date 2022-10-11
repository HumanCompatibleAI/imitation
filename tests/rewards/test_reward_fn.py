"""Tests `imitation.rewards.reward_function` and `imitation.rewards.serialize`."""

import numpy as np
import pytest

from imitation.rewards import reward_function, serialize

OBS = np.random.randint(0, 10, (64, 100))
ACTS = NEXT_OBS = OBS
DONES = np.zeros(64, dtype=np.bool_)


def _funky_reward_fn(obs, act, next_obs, done):
    """Returns consecutive reward from 1 to batch size `len(obs)`."""
    # give each environment number from 1 to num_envs
    return (np.arange(len(obs))).astype("float32")


def _invalid_reward_fn(obs, act, next_obs, done):
    """Returns rewards for lesser number of observations."""
    return (np.arange(len(obs) - 1)).astype("float32")


def test_reward_fn_override():
    # test inheriting class from RewardFn works
    class InheritedFunkyReward(reward_function.RewardFn):
        """A reward inherited from RewardFn."""

        def __init__(self):
            super().__init__()

        def __call__(self, obs, act, next_obs, steps=None):
            """Returns consecutive reward from 0 to batch size -1 (`len(obs)` - 1)."""
            return (np.arange(len(obs))).astype("float32")

    inherited_funky_reward_fn = InheritedFunkyReward()
    inherited_funky_reward_fn(OBS, ACTS, NEXT_OBS)


def test_validate_rewardfn_class():
    validated_reward_fn = serialize.ValidateRewardFn(_funky_reward_fn)
    validated_reward_fn(OBS, ACTS, NEXT_OBS, DONES)

    with pytest.raises(AssertionError):
        invalidated_reward_fn = serialize.ValidateRewardFn(_invalid_reward_fn)
        invalidated_reward_fn(OBS, ACTS, NEXT_OBS, DONES)
