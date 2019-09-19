"""Test imitation.envs.*."""

import gym
import pytest

# Unused imports is for side-effect of registering environments
from imitation.envs import examples  # noqa: F401
from imitation.testing import envs

ENV_NAMES = [env_spec.id for env_spec in gym.envs.registration.registry.all()
             if env_spec.id.startswith('imitation/')]

DETERMINISTIC_ENVS = []


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
class TestEnvs:
  """Battery of simple tests for environments."""
  def test_seed(self, env, env_name):
    envs.test_seed(env, env_name, DETERMINISTIC_ENVS)

  def test_premature_step(self, env):
    """Test that you must call reset() before calling step()."""
    if hasattr(env, 'sim') and hasattr(env, 'model'):  # pragma: no cover
      # We can't use isinstance since importing mujoco_py will fail on
      # machines without MuJoCo installed
      pytest.skip("MuJoCo environments cannot perform this check.")

    act = env.action_space.sample()
    with pytest.raises(Exception):  # need to call env.reset() first
      env.step(act)

  def test_model_based(self, env):
    """Smoke test for each of the ModelBasedEnv methods with type checks."""
    if not hasattr(env, 'state_space'):  # pragma: no cover
      pytest.skip("This test is only for subclasses of ModelBasedEnv.")

    envs.test_model_based(env)

  def test_rollout(self, env):
    envs.test_rollout(env)
