"""Tests interactive policies."""

import collections
from unittest import mock

import gym
import numpy as np
import pytest
from stable_baselines3.common import vec_env

from imitation.policies import interactive

ENVS = [
    "CartPole-v0",
]


class NoRenderingDiscreteInteractivePolicy(interactive.DiscreteInteractivePolicy):
    """DiscreteInteractivePolicy with no rendering."""

    def _render(self, obs: np.ndarray) -> None:
        pass


def _get_interactive_policy(env: vec_env.VecEnv):
    num_actions = env.envs[0].action_space.n
    action_keys_names = collections.OrderedDict(
        [(f"k{i}", f"n{i}") for i in range(num_actions)]
    )
    interactive_policy = NoRenderingDiscreteInteractivePolicy(
        env.observation_space, env.action_space, action_keys_names
    )
    return interactive_policy


@pytest.mark.parametrize("env_name", ENVS)
def test_interactive_policy(env_name: str):
    """Test if correct actions are selected, as specified by input keys."""
    env = vec_env.DummyVecEnv([lambda: gym.wrappers.TimeLimit(gym.make(env_name), 10)])
    env.seed(0)

    interactive_policy = _get_interactive_policy(env)
    action_keys = list(interactive_policy.action_keys_names.keys())

    obs = env.reset()
    done = np.array([False])

    def mock_input(_):
        # Sometimes insert incorrect keys, which should get ignored by the policy.
        if np.random.uniform() < 0.5:
            return "invalid"
        key = action_keys[mock_input.index]
        mock_input.index = (mock_input.index + 1) % len(action_keys)
        return key

    mock_input.index = 0

    with mock.patch("builtins.input", mock_input):
        requested_action = 0
        while not done.all():
            action, _ = interactive_policy.predict(obs)
            assert isinstance(action, np.ndarray)
            assert all(env.action_space.contains(a) for a in action)
            assert action[0] == requested_action

            obs, reward, done, info = env.step(action)
            assert isinstance(obs, np.ndarray)
            assert all(env.observation_space.contains(o) for o in obs)
            assert isinstance(reward, np.ndarray)
            assert isinstance(done, np.ndarray)

            requested_action = (requested_action + 1) % len(action_keys)


@pytest.mark.parametrize("env_name", ENVS)
def test_interactive_policy_input_validity(capsys, env_name: str):
    """Test if appropriate feedback is given on the validity of the input."""
    env = vec_env.DummyVecEnv([lambda: gym.wrappers.TimeLimit(gym.make(env_name), 10)])
    env.seed(0)

    interactive_policy = _get_interactive_policy(env)
    action_keys = list(interactive_policy.action_keys_names.keys())

    # Valid input key case
    obs = env.reset()

    def mock_input_valid(prompt):
        print(prompt)
        return action_keys[0]

    with mock.patch("builtins.input", mock_input_valid):
        interactive_policy.predict(obs)
        stdout = capsys.readouterr().out
        assert "Your choice" in stdout and "Invalid" not in stdout

    # First invalid input key, then valid
    obs = env.reset()

    def mock_input_invalid_then_valid(prompt):
        print(prompt)
        if mock_input_invalid_then_valid.return_valid:
            return action_keys[0]
        mock_input_invalid_then_valid.return_valid = True
        return "invalid"

    mock_input_invalid_then_valid.return_valid = False

    with mock.patch("builtins.input", mock_input_invalid_then_valid):
        interactive_policy.predict(obs)
        stdout = capsys.readouterr().out
        assert "Your choice" in stdout and "Invalid" in stdout
