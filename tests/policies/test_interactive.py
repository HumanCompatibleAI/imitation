"""Tests interactive policies."""

import collections
from unittest import mock

import gym
import numpy as np
import pytest
from stable_baselines3.common import vec_env

from imitation.policies import interactive

SIMPLE_ENVS = [
    "seals/CartPole-v0",
]
ATARI_ENVS = [
    "Pong-v4",
]


class NoRenderingDiscreteInteractivePolicy(interactive.DiscreteInteractivePolicy):
    """DiscreteInteractivePolicy with no rendering."""

    def _render(self, obs: np.ndarray) -> None:
        pass


def _get_simple_interactive_policy(env: vec_env.VecEnv):
    num_actions = env.action_space.n
    action_keys_names = collections.OrderedDict(
        [(f"k{i}", f"n{i}") for i in range(num_actions)],
    )
    interactive_policy = NoRenderingDiscreteInteractivePolicy(
        env.observation_space,
        env.action_space,
        action_keys_names,
    )
    return interactive_policy


@pytest.mark.parametrize("env_name", SIMPLE_ENVS + ATARI_ENVS)
def test_interactive_policy(env_name: str):
    """Test if correct actions are selected, as specified by input keys."""
    env = vec_env.DummyVecEnv([lambda: gym.wrappers.TimeLimit(gym.make(env_name), 50)])
    env.seed(0)

    if env_name in ATARI_ENVS:
        interactive_policy = interactive.AtariInteractivePolicy(env)
    else:
        interactive_policy = _get_simple_interactive_policy(env)
    action_keys = list(interactive_policy.action_keys_names.keys())

    obs = env.reset()
    done = np.array([False])

    class mock_input:
        def __init__(self):
            self.index = 0

        def __call__(self, _):
            # Sometimes insert incorrect keys, which should get ignored by the policy.
            if np.random.uniform() < 0.5:
                return "invalid"
            key = action_keys[self.index]
            self.index = (self.index + 1) % len(action_keys)
            return key

    with mock.patch("builtins.input", mock_input()):
        requested_action = 0
        while not done.all():
            action, _ = interactive_policy.predict(np.array(obs))
            assert isinstance(action, np.ndarray)
            assert all(env.action_space.contains(a) for a in action)
            assert action[0] == requested_action

            obs, reward, done, info = env.step(action)
            assert isinstance(obs, np.ndarray)
            assert all(env.observation_space.contains(o) for o in obs)
            assert isinstance(reward, np.ndarray)
            assert isinstance(done, np.ndarray)

            requested_action = (requested_action + 1) % len(action_keys)


@pytest.mark.parametrize("env_name", SIMPLE_ENVS)
def test_interactive_policy_input_validity(capsys, env_name: str):
    """Test if appropriate feedback is given on the validity of the input."""
    env = vec_env.DummyVecEnv([lambda: gym.wrappers.TimeLimit(gym.make(env_name), 10)])
    env.seed(0)

    interactive_policy = _get_simple_interactive_policy(env)
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

    class mock_input_invalid_then_valid:
        def __init__(self):
            self.return_valid = False

        def __call__(self, prompt):
            print(prompt)
            if self.return_valid:
                return action_keys[0]
            self.return_valid = True
            return "invalid"

    with mock.patch("builtins.input", mock_input_invalid_then_valid()):
        interactive_policy.predict(obs)
        stdout = capsys.readouterr().out
        assert "Your choice" in stdout and "Invalid" in stdout


@pytest.mark.parametrize("env_name", ATARI_ENVS)
def test_atari_action_mappings(env_name: str):
    """Test if correct actions are selected, as specified by input keys."""
    env = vec_env.DummyVecEnv([lambda: gym.wrappers.TimeLimit(gym.make(env_name), 50)])
    env.seed(0)
    action_meanings = env.env_method("get_action_meanings", indices=[0])[0]

    interactive_policy = interactive.AtariInteractivePolicy(env)

    obs = env.reset()

    provided_keys = ["2", "a", "d"]
    expected_action_meanings = ["FIRE", "LEFT", "RIGHT"]

    class mock_input:
        def __init__(self):
            self.index = 0

        def __call__(self, _):
            key = provided_keys[self.index]
            self.index += 1
            return key

    with mock.patch("builtins.input", mock_input()):
        for expected_action_meaning in expected_action_meanings:
            action, _ = interactive_policy.predict(np.array(obs))
            obs, reward, done, info = env.step(action)

            assert action_meanings[action[0]] == expected_action_meaning
