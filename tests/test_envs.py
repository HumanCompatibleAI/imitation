import gym
import pytest

# Import for side-effect of registering environment
import imitation.examples.airl_envs  # noqa: F401
import imitation.examples.model_envs  # noqa: F401
import imitation.examples.env_suite # noqa: F401

ENV_NAMES = [env_spec.id for env_spec in gym.envs.registration.registry.all()
             if env_spec.id.startswith('imitation/')]


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_envs(env_name):
  """Check that our custom environments don't crash on `step`, and `reset`."""
  try:
    env = gym.make(env_name)
  except gym.error.DependencyNotInstalled as e:  # pragma: no cover
    if e.args[0].find('mujoco_py') != -1:
      pytest.skip("Requires `mujoco_py`, which isn't installed.")
    else:
      raise
  env.reset()
  obs_space = env.observation_space
  for _ in range(4):
    act = env.action_space.sample()
    obs, rew, done, info = env.step(act)
    assert obs in obs_space
