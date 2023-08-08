"""Tests InteractivePolicy."""
import random
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
from stable_baselines3.common.vec_env import VecEnv

from imitation.policies.interactive import InteractivePolicy, query_human
from imitation.util.util import make_vec_env

_make_vec_env = partial(make_vec_env, n_envs=1, rng=np.random.default_rng(42))


ENVS = [
    pytest.param(_make_vec_env("FrozenLake-v1"), id="FrozenLake-v1"),
]


def get_interactive_policy(env: VecEnv):
    def query_fn(obs):
        env.render()
        return query_human()

    return InteractivePolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        query_fn=query_fn,
    )


@pytest.mark.parametrize("env", ENVS)
def test_interactive_policy_valid(env: VecEnv):
    interactive_policy = get_interactive_policy(env)
    obs = env.reset()
    done = np.array([False])

    def mock_input_valid(_):
        return random.choice(["w", "a", "s", "d"])

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


@pytest.mark.parametrize("env", ENVS)
def test_interactive_policy_invalid(capsys, env: VecEnv):
    interactive_policy = get_interactive_policy(env)
    obs = env.reset()

    def mock_input_invalid(_):
        return random.choice(["x", "y", "z"])

    with patch("builtins.input", mock_input_invalid):
        with pytest.raises(ValueError):
            action, _ = interactive_policy.predict(obs)
