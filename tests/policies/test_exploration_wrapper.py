"""Tests ExplorationWrapper."""

import numpy as np
import pytest
import seals  # noqa: F401

from imitation.policies import exploration_wrapper
from imitation.util import util


def constant_policy(obs, state, mask):
    del state, mask
    return np.zeros(len(obs), dtype=int), None


def fake_stateful_policy(obs, state, mask):
    del state, mask
    return np.zeros(len(obs), dtype=int), (np.zeros(1),)


def make_wrapper(random_prob, switch_prob, rng):
    venv = util.make_vec_env(
        "seals/CartPole-v0",
        n_envs=1,
        rng=rng,
    )
    return (
        exploration_wrapper.ExplorationWrapper(
            policy=constant_policy,
            venv=venv,
            random_prob=random_prob,
            switch_prob=switch_prob,
            rng=rng,
        ),
        venv,
    )


def test_random_prob(rng):
    """Test that `random_prob` produces right behaviors of policy switching.

    The policy always makes an initial switch when ExplorationWrapper is applied.
    Then the policy makes switches according to `switch_prob`.

    This test fixes `switch_prob` to 0.5 and sets `random_prob` to 0.0, 1.0 and 0.5.
    (1) `random_prob=0.0`: Initial and following policies are always constant policies.
    (2) `random_prob=1.0`: Initial and following policies are always random policies.
    (3) `random_prob=0.5`: Around half-half for constant and random policies.

    Args:
        rng (np.random.Generator): random number generator.

    Raises:
        ValueError: Unknown policy type to switch.
    """
    wrapper, _ = make_wrapper(random_prob=0.0, switch_prob=0.5, rng=rng)
    assert wrapper.current_policy == constant_policy
    for _ in range(100):
        wrapper._switch()
        assert wrapper.current_policy == constant_policy

    wrapper, _ = make_wrapper(random_prob=1.0, switch_prob=0.5, rng=rng)
    assert wrapper.current_policy == wrapper._random_policy
    for _ in range(100):
        wrapper._switch()
        assert wrapper.current_policy == wrapper._random_policy

    wrapper, _ = make_wrapper(random_prob=0.5, switch_prob=0.5, rng=rng)
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


def test_switch_prob(rng):
    """Test that `switch_prob` produces right behaviors of policy switching.

    The policy always makes an initial switch when ExplorationWrapper is applied.
    Then the policy makes switches according to `switch_prob`.

    This test set includes the following:
    (1) `switch_prob=0.0`: The policy never switches after initial switch.
    (2) `switch_prob=1.0`: The policy always switches and the distribution of
        policies is determined by `random_prob`.

    Args:
        rng (np.random.Generator): random number generator.
    """
    wrapper, venv = make_wrapper(random_prob=0.5, switch_prob=0.0, rng=rng)
    policy = wrapper.current_policy

    obs = np.random.rand(100, 2)
    for action in wrapper(obs, None, None)[0]:
        assert venv.action_space.contains(action)
        assert wrapper.current_policy == policy

    def _always_switch(random_prob, num_steps):
        wrapper, _ = make_wrapper(random_prob=random_prob, switch_prob=1.0, rng=rng)
        num_random = 0
        num_constant = 0
        for _ in range(num_steps):
            obs = np.random.rand(1, 2)
            wrapper(obs, None, None)
            if wrapper.current_policy == wrapper._random_policy:
                num_random += 1
            elif wrapper.current_policy == constant_policy:
                num_constant += 1
            else:  # pragma: no cover
                raise ValueError("Unknown policy")
        return num_random, num_constant

    num_random, num_constant = _always_switch(
        random_prob=1.0,
        num_steps=5000,
    )
    assert num_random == 5000
    assert num_constant == 0
    num_random, num_constant = _always_switch(
        random_prob=0.5,
        num_steps=5000,
    )
    assert num_random > 2250
    assert num_constant > 2250
    num_random, num_constant = _always_switch(
        random_prob=0.0,
        num_steps=5000,
    )
    assert num_random == 0
    assert num_constant == 5000


def test_valid_output(rng):
    """Ensure that we test both the random and the wrapped policy at least once."""
    for random_prob in [0.0, 0.5, 1.0]:
        wrapper, venv = make_wrapper(random_prob=random_prob, switch_prob=0.5, rng=rng)
        np.random.seed(0)
        obs = np.random.rand(100, 2)
        for action in wrapper(obs, None, None)[0]:
            assert venv.action_space.contains(action)


def test_throws_for_stateful_policy(rng):
    venv = util.make_vec_env(
        "seals/CartPole-v0",
        n_envs=1,
        rng=rng,
    )
    wrapper = exploration_wrapper.ExplorationWrapper(
        policy=fake_stateful_policy,
        venv=venv,
        random_prob=0,
        switch_prob=0,
        rng=rng,
    )

    np.random.seed(0)
    obs = np.random.rand(100, 2)
    with pytest.raises(
        ValueError,
        match="Exploration wrapper does not support stateful policies.",
    ):
        wrapper(obs, (np.ones_like(obs)), None)

    with pytest.raises(
        ValueError,
        match="Exploration wrapper does not support stateful policies.",
    ):
        wrapper(obs, None, None)
