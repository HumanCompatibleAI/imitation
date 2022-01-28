"""Tests ExplorationWrapper."""

import numpy as np
import seals  # noqa: F401

from imitation.policies import exploration_wrapper
from imitation.util import util


def constant_policy(obs):
    return np.zeros(len(obs), dtype=int)


def make_wrapper(random_prob, switch_prob):
    venv = util.make_vec_env(
        "seals/CartPole-v0",
        n_envs=1,
    )
    return (
        exploration_wrapper.ExplorationWrapper(
            policy=constant_policy,
            venv=venv,
            random_prob=random_prob,
            switch_prob=switch_prob,
            seed=0,
        ),
        venv,
    )


def test_switch():
    wrapper, _ = make_wrapper(random_prob=0.0, switch_prob=0.5)
    assert wrapper.current_policy == constant_policy
    for _ in range(100):
        wrapper._switch()
        assert wrapper.current_policy == constant_policy

    wrapper, _ = make_wrapper(random_prob=1.0, switch_prob=0.5)
    assert wrapper.current_policy == wrapper._random_policy
    for _ in range(100):
        wrapper._switch()
        assert wrapper.current_policy == wrapper._random_policy

    wrapper, _ = make_wrapper(random_prob=0.5, switch_prob=0.5)
    num_random = 0
    num_constant = 0
    for _ in range(1000):
        wrapper._switch()
        if wrapper.current_policy == wrapper._random_policy:
            num_random += 1
        elif wrapper.current_policy == constant_policy:
            num_constant += 1
        else:  # pragma: no cover
            raise ValueError("Unknown policy")
    # Holds with very high probability (and seeding means it's deterministic)
    assert num_random > 450
    assert num_constant > 450


def test_valid_output():
    # Ensure that we test both the random and the wrapped policy
    # at least once:
    for random_prob in [0.0, 0.5, 1.0]:
        wrapper, venv = make_wrapper(random_prob=random_prob, switch_prob=0.5)
        np.random.seed(0)
        obs = np.random.rand(100, 2)
        for action in wrapper(obs):
            assert venv.action_space.contains(action)
