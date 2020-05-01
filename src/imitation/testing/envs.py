"""Helper methods for tests of custom Gym environments.

This is used in the imitation test suite and may also be useful for users of this
library.
"""

import gym


def test_model_based(env: gym.Env) -> None:
    """Smoke test for each of the ModelBasedEnv methods with type checks.

    Raises:
        AssertionError if test fails.
    """
    state = env.initial_state()
    assert env.state_space.contains(state)

    action = env.action_space.sample()
    new_state = env.transition(state, action)
    assert env.state_space.contains(new_state)

    reward = env.reward(state, action, new_state)
    assert isinstance(reward, float)

    done = env.terminal(state, 0)
    assert isinstance(done, bool)

    obs = env.obs_from_state(state)
    assert env.observation_space.contains(obs)
    next_obs = env.obs_from_state(new_state)
    assert env.observation_space.contains(next_obs)
