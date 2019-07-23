import gym
import pytest

from imitation.examples.airl_envs import ENV_NAMES

PARALLEL = [False, True]

try:  # pragma: no cover
  import mujoco_py as _  # pytype: disable=import-error
  del _
  MUJOCO_OK = True
except ImportError:
  MUJOCO_OK = False


@pytest.mark.skipif(not MUJOCO_OK,
                    reason="Requires `mujoco_py`, which isn't installed.")
@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_envs(env_name):  # pragma: no cover
  """Check that our custom environments don't crash on `step`, and `reset`."""
  for env_name in ENV_NAMES:
    env = gym.make(env_name)
    env.reset()
    obs_space = env.observation_space
    for _ in range(4):
      act = env.action_space.sample()
      obs, rew, done, info = env.step(act)
      assert obs in obs_space
