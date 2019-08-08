import re

import gym
import numpy as np
import pytest

# Unused imports are for side-effect of registering environments
import imitation.examples.airl_envs  # noqa: F401
import imitation.examples.model_envs  # noqa: F401
import imitation.model_env

ENV_NAMES = [env_spec.id for env_spec in gym.envs.registration.registry.all()
             if env_spec.id.startswith('imitation/')]

DETERMINISTIC_ENVS = [
    r"imitation/CliffWorld.*-v0",
    r"imitation/PointMaze.*-v0",
]


def matches_list(env_name, patterns):
  for pattern in patterns:
    if re.compile(pattern).match(env_name):
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


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_seed(env, env_name):
  # With the same seed, should always give the same result
  seeds = env.seed(42)
  assert isinstance(seeds, list)
  assert len(seeds) > 0

  first_obs = env.reset()
  env.seed(42)
  second_obs = env.reset()
  assert np.all(first_obs == second_obs)

  # For non-deterministic environments, if we try enough seeds we should
  # eventually get a different result. For deterministic environments, all
  # seeds will produce the same starting state.
  same_obs = True
  for i in range(10):
    env.seed(i)
    obs = env.reset()
    if not np.all(obs == first_obs):
      same_obs = False
      break

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
  if not isinstance(env, imitation.model_env.ModelBasedEnv):
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
