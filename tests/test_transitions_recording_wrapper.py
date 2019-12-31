import gym
import numpy as np
import pytest
from stable_baselines.common.vec_env import DummyVecEnv

from imitation.util.transitions_recording_wrapper import \
    TransitionsRecordingWrapper


class CountingEnv(gym.Env):
  """At timestep `t`, has `reward == obs == t`."""

  def __init__(self):
    self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=())
    self.action_space = gym.spaces.Box(low=0, high=np.inf, shape=())
    self.timestep = None

  def reset(self):
    self.timestep = 0

  def step(self, action):
    assert self.timestep is not None
    assert action in self.action_space
    return self.timestep, self.timestep, False, {}


def _make_recording_venv(error_on_premature_reset: bool,
                         ) -> TransitionsRecordingWrapper:
  venv = DummyVecEnv([CountingEnv] * 2)
  venv = TransitionsRecordingWrapper(venv, error_on_premature_reset)
  venv.reset()
  return venv


def test_buffer_wrapper_pop_transitions():
  """Check transitions_pop() result for BufferWrapper.

  To make things easy to check, we use a dummy environment where the observation
  is simply the timestep.
  """
  venv = _make_recording_venv
  for _ in range(3):
    expect_obs_list = [0]
    expect_acts_list = []

    obs = venv.reset()
    np.testing.assert_array_equal(obs, [0, 0])
    expect_obs_list.append(obs)

    for t in range(1, 10):
      acts = np.random.random(size=(2,))
      expect_acts_list.append(acts)

      obs, *_ = venv.step(acts)
      np.testing.assert_array_equal(obs, [t, t])
      expect_obs_list.append(obs)

    samples = venv.pop_transitions()
    np.testing.assert_array_equal(samples.obs, expect_obs_list[:-1])
    np.testing.assert_array_equal(samples.acts, expect_acts_list)
    np.testing.assert_array_equal(samples.new_obs, expect_obs_list[1:])


def test_vec_env_recording_error():
  # Resetting before a `step()` is okay.
  for flag in [True, False]:
    venv = _make_recording_venv(flag)
    for _ in range(10):
      venv.reset()

  # Resetting after a `step()` is not okay if error flag is True.
  venv = _make_recording_venv(True)
  venv.step()
  with pytest.raises(RuntimeError, matches="TransitionsRecordingWrapper"):
    venv.reset()

  # Same as previous case, but insert a `pop_transitions()` in between.
  venv = _make_recording_venv(True)
  venv.step()
  venv.pop_transitions()
  venv.step()
  with pytest.raises(RuntimeError, matches="TransitionsRecordingWrapper"):
    venv.reset()

  # Resetting after a `step()` is ok if error flag is False.
  venv = _make_recording_venv(False)
  venv.step()
  venv.reset()

  # Resetting after a `step()` is ok if transitions are first collected.
  for flag in [True, False]:
    venv = _make_recording_venv(flag)
    venv.step()
    venv.pop_transitions()
    venv.reset()
