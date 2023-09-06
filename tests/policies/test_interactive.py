"""Tests interactive policies."""
import random
from unittest.mock import patch

import gym
import numpy as np
import pytest
from stable_baselines3.common import vec_env

from imitation.policies import interactive

ENVS = [
    "Pong-v4",
]


class NoRenderingDiscreteInteractivePolicy(interactive.DiscreteInteractivePolicy):
    def _render(self, obs: np.ndarray) -> None:
        pass


@pytest.mark.parametrize("env_name", ENVS)
def test_interactive_policy(env_name: str):
    env = vec_env.DummyVecEnv([lambda: gym.wrappers.TimeLimit(gym.make(env_name), 10)])
    env.seed(0)

    num_actions = env.envs[0].action_space.n
    action_names = [f"n{i}" for i in range(num_actions)]
    action_keys = [f"k{i}" for i in range(num_actions)]
    interactive_policy = NoRenderingDiscreteInteractivePolicy(
        env.observation_space,
        env.action_space,
        action_names,
        action_keys,
    )

    obs = env.reset()
    done = np.array([False])

    def mock_input_valid(_):
        return random.choice(action_keys)

    with patch("builtins.input", mock_input_valid):
        while not done.all():
            action, _ = interactive_policy.predict(obs)
            assert isinstance(action, np.ndarray)
            assert all(env.action_space.contains(a) for a in action)

            obs, reward, done, info = env.step(action)
            assert isinstance(obs, np.ndarray)
            assert all(env.observation_space.contains(o) for o in obs)
            assert isinstance(reward, np.ndarray)
            assert isinstance(done, np.ndarray)
