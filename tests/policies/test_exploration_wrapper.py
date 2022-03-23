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
            policy_callable=constant_policy,
            venv=venv,
            random_prob=random_prob,
            switch_prob=switch_prob,
            seed=0,
        ),
        venv,
    )


def test_random_prob():
    """Test that `random_prob` produces right behaviors of policy switching.

    The policy always makes an initial switch when ExplorationWrapper is applied.
    Then the policy makes switches according to `switch_prob`.

    This test fixes `switch_prob` to 0.5 and sets `random_prob` to 0.0, 1.0 and 0.5.
    (1) `random_prob=0.0`: Initial and following policies are always constant policies.
    (2) `random_prob=1.0`: Initial and following policies are always random policies.
    (3) `random_prob=0.5`: Around half-half for constant and random policies.

    Raises:
        ValueError: Unknown policy type to switch.
    """
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


def test_switch_prob():
    """Test that `switch_prob` produces right behaviors of policy switching.

    The policy always makes an initial switch when ExplorationWrapper is applied.
    Then the policy makes switches according to `switch_prob`.

    This test set includes the following:
    (1) `switch_prob=0.0`: The policy never switches after initial switch.
    (2) `switch_prob=1.0`: The policy always switches and the distribution of
        policies is determined by `random_prob`.
    """
    wrapper, venv = make_wrapper(random_prob=0.5, switch_prob=0.0)
    policy = wrapper.current_policy
    np.random.seed(0)
    obs = np.random.rand(100, 2)
    for action in wrapper(obs):
        assert venv.action_space.contains(action)
        assert wrapper.current_policy == policy

    def _always_switch(random_prob, num_steps, seed):
        wrapper, _ = make_wrapper(random_prob=random_prob, switch_prob=1.0)
        np.random.seed(seed)
        num_random = 0
        num_constant = 0
        for _ in range(num_steps):
            obs = np.random.rand(1, 2)
            wrapper(obs)
            if wrapper.current_policy == wrapper._random_policy:
                num_random += 1
            elif wrapper.current_policy == constant_policy:
                num_constant += 1
            else:  # pragma: no cover
                raise ValueError("Unknown policy")
        return num_random, num_constant

    num_random, num_constant = _always_switch(random_prob=1.0, num_steps=1000, seed=0)
    assert num_random == 1000
    assert num_constant == 0
    num_random, num_constant = _always_switch(random_prob=0.5, num_steps=1000, seed=0)
    assert num_random > 450
    assert num_constant > 450
    num_random, num_constant = _always_switch(random_prob=0.0, num_steps=1000, seed=0)
    assert num_random == 0
    assert num_constant == 1000


def test_valid_output():
    """Ensure that we test both the random and the wrapped policy at least once."""
    for random_prob in [0.0, 0.5, 1.0]:
        wrapper, venv = make_wrapper(random_prob=random_prob, switch_prob=0.5)
        np.random.seed(0)
        obs = np.random.rand(100, 2)
        for action in wrapper(obs):
            assert venv.action_space.contains(action)
