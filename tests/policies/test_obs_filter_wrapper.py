"""Tests for `imitation.policies.obs_filter_wrapper`."""

import gymnasium as gym
import numpy as np
from stable_baselines3.common import torch_layers

from imitation.data.wrappers import HumanReadableWrapper
from imitation.policies.obs_filter_wrapper import RemoveHR
from imitation.policies import base as policy_base


def test_remove_hr():
    env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
    hr_env = HumanReadableWrapper(env)
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

    obs_with_hr, _ = hr_env.reset(seed=0)
    pred_action_with_hr, _ = wrapped_policy.predict(obs_with_hr, deterministic=True)
    assert np.equal(pred_action, pred_action_with_hr).all()
