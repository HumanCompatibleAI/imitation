"""Tests for `imitation.policies.obs_update_wrapper`."""

from typing import Dict

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common import torch_layers

from imitation.data.wrappers import HR_OBS_KEY, HumanReadableWrapper
from imitation.policies import base as policy_base
from imitation.policies.obs_update_wrapper import RemoveHR, _remove_hr_obs


@pytest.mark.parametrize("use_hr_wrapper", [True, False])
def test_remove_hr(use_hr_wrapper: bool):
    env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
    new_env = HumanReadableWrapper(env) if use_hr_wrapper else env
    policy = policy_base.FeedForward32Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 1.0,
        features_extractor_class=torch_layers.FlattenExtractor,
    )
    wrapped_policy = RemoveHR(policy, lr_schedule=lambda _: 1.0)
    assert wrapped_policy.net_arch == policy.net_arch

    obs, _ = env.reset(seed=0)
    pred_action, _ = wrapped_policy.predict(obs, deterministic=True)

    new_obs, _ = new_env.reset(seed=0)
    pred_action_with_hr, _ = wrapped_policy.predict(new_obs, deterministic=True)
    assert np.equal(pred_action, pred_action_with_hr).all()


@pytest.mark.parametrize(
    ("testname", "obs", "expected_obs"),
    [
        (
            "np.ndarray",
            np.array([1]),
            np.array([1]),
        ),
        (
            "dict with np.ndarray",
            {"a": np.array([1])},
            {"a": np.array([1])},
        ),
        (
            "dict rgb removed successfully and got unwrapped from dict",
            {
                "a": np.array([1]),
                HR_OBS_KEY: np.array([3]),
            },
            np.array([1]),
        ),
        (
            "dict rgb removed successfully and got dict",
            {
                "a": np.array([1]),
                "b": np.array([2]),
                HR_OBS_KEY: np.array([3]),
            },
            {
                "a": np.array([1]),
                "b": np.array([2]),
            },
        ),
    ],
)
def test_remove_rgb_ob(testname, obs, expected_obs):
    got_obs = _remove_hr_obs(obs)
    assert type(got_obs) is type(expected_obs)
    if isinstance(got_obs, (Dict, gym.spaces.Dict)):
        assert len(got_obs.keys()) == len(expected_obs.keys())
        for k, v in got_obs.items():
            assert v == expected_obs[k]
    else:
        assert got_obs == expected_obs


def test_remove_rgb_obs_failure():
    with pytest.raises(ValueError, match="Only human readable observation*"):
        _remove_hr_obs({HR_OBS_KEY: np.array([1])})


def test_remove_rgb_obs_still_keep_origin_space_rgb():
    obs = {"a": np.array([1]), HR_OBS_KEY: np.array([2])}
    _remove_hr_obs(obs)
    assert HR_OBS_KEY in obs
