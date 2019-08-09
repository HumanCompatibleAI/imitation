import re

import gym
import numpy as np
import pytest

# Unused imports are for side-effect of registering environments
import imitation.examples.airl_envs  # noqa: F401
import imitation.examples.model_envs  # noqa: F401

ENV_NAMES = [env_spec.id for env_spec in gym.envs.registration.registry.all()
             if env_spec.id.startswith('imitation/')]

DETERMINISTIC_ENVS = []


def matches_list(env_name, patterns):
  for pattern in patterns:  # pragma: no cover
    if re.match(pattern, env_name):
      return True
  return False


@pytest.fixture
def env(env_name):
  try:
    env = gym.make(env_name)
  except gym.error.DependencyNotInstalled as e:  # pragma: no cover
    if e.args[0].find('mujoco_py') != -1:
      pytest.skip("Requires `mujoco_py`, which isn't installed.")
    else:
      raise
  return env


def rollout(env, actions):
  ret = [(env.reset(), None, None, None)]
  for act in actions:
    ret.append(env.step(act))
  return ret


def assert_equal_rollout(rollout_a, rollout_b):
  for step_a, step_b in zip(rollout_a, rollout_b):
    ob_a, rew_a, done_a, info_a = step_a
    ob_b, rew_b, done_b, info_b = step_b
    assert np.all(ob_a == ob_b)
    assert rew_a == rew_b
    assert done_a == done_b
    assert info_a == info_b


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_seed(env, env_name):
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
  for i in range(10):
    env.seed(i)
    new_rollout = rollout(env, actions)
    for step_a, step_new in zip(rollout_a, new_rollout):
      obs_a = step_a[0]
      obs_new = step_new[0]
      if np.any(obs_a != obs_new):
        same_obs = False

  is_deterministic = matches_list(env_name, DETERMINISTIC_ENVS)
  assert same_obs == is_deterministic


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_premature_step(env):
  if hasattr(env, 'sim') and hasattr(env, 'model'):  # pragma: no cover
    # We can't use isinstance since importing mujoco_py will fail on
    # machines without MuJoCo installed
    pytest.skip("MuJoCo environments cannot perform this check.")

  act = env.action_space.sample()
  with pytest.raises(Exception):  # need to call env.reset() first
    env.step(act)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_model_based(env):
  """Smoke test for each of the ModelBasedEnv methods with basic type checks."""
  if not hasattr(env, 'state_space'):  # pragma: no cover
    pytest.skip("This test is only for subclasses of ModelBasedEnv.")

  old_state = env.initial_state()
  assert env.state_space.contains(old_state)

  action = env.action_space.sample()
  new_state = env.transition(old_state, action)
  assert env.state_space.contains(new_state)

  reward = env.reward(old_state, action, new_state)
  assert isinstance(reward, float)

  done = env.terminal(old_state, 0)
  assert isinstance(done, bool)

  old_obs = env.obs_from_state(old_state)
  assert env.observation_space.contains(old_obs)


@pytest.mark.parametrize("env_name", ENV_NAMES)
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
