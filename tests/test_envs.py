import gym
import pytest

from imitation.envs import ENV_NAMES
from imitation.util import make_vec_env

PARALLEL = [False, True]


@pytest.mark.parametrize("parallel", PARALLEL)
def test_envs(parallel):
  """Check that our custom environments don't crash on `step`, and `reset`."""
  for env_name in ENV_NAMES:
    if parallel:
      env = make_vec_env(env_name)
    else:
      env = gym.make(env_name)

      env.reset()
      obs_space = env.observation_space
      for _ in range(4):
        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)
        assert obs in obs_space
