"""Helper methods for tests of custom Gym environments.

This is used in the imitation test suite and may also be useful for users
of this library.
"""

import re

import numpy as np


def matches_list(env_name, patterns):
  for pattern in patterns:  # pragma: no cover
    if re.match(pattern, env_name):
      return True
  return False


def rollout(env, actions):
  ret = [(env.reset(), None, None, None)]
  for act in actions:
    ret.append(env.step(act))
  return ret


def assert_equal_rollout(rollout_a, rollout_b):
  for step_a, step_b in zip(rollout_a, rollout_b):
    ob_a, rew_a, done_a, info_a = step_a
    ob_b, rew_b, done_b, info_b = step_b
    np.testing.assert_equal(ob_a, ob_b)
    assert rew_a == rew_b
    assert done_a == done_b
    np.testing.assert_equal(info_a, info_b)


def test_seed(env, env_name, deterministic_envs):
  """Tests environment seeding.

  If non-deterministic, different seeds should produce different transitions.
  If deterministic, should be invariant to seed.
  """
  env.action_space.seed(0)
  actions = [env.action_space.sample() for _ in range(10)]

  # With the same seed, should always get the same result
  seeds = env.seed(42)
  assert isinstance(seeds, list)
  assert len(seeds) > 0
  rollout_a = rollout(env, actions)

  env.seed(42)
  rollout_b = rollout(env, actions)

  assert_equal_rollout(rollout_a, rollout_b)

  # For non-deterministic environments, if we try enough seeds we should
  # eventually get a different result. For deterministic environments, all
  # seeds will produce the same starting state.
  same_obs = True
  for i in range(20):
    env.seed(i)
    new_rollout = rollout(env, actions)
    for step_a, step_new in zip(rollout_a, new_rollout):
      obs_a = step_a[0]
      obs_new = step_new[0]
      if np.any(obs_a != obs_new):
        same_obs = False
        break
    if not same_obs:
      break

  is_deterministic = matches_list(env_name, deterministic_envs)
  assert same_obs == is_deterministic


def test_model_based(env):
  """Smoke test for each of the ModelBasedEnv methods with type checks."""
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


def test_rollout(env):
  """Check custom environments have correct types on `step` and `reset`."""
  obs_space = env.observation_space
  obs = env.reset()
  assert obs in obs_space

  for _ in range(4):
    act = env.action_space.sample()
    obs, rew, done, info = env.step(act)
    assert obs in obs_space
    assert isinstance(rew, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
