from typing import Sequence

import gym
import numpy as np
import pytest
from stable_baselines.common.vec_env import DummyVecEnv

from imitation.util import rollout
from imitation.util.buffering_wrapper import BufferingWrapper


class _CountingEnv(gym.Env):
  """At timestep `t` of each episode, has `reward / 10 == obs == t`.

  Episodes finish after `episode_length` calls to `step()`. As an example,
  if we have `episode_length=5`, then an episode is to have the
  following observations and rewards:

  ```
  obs = [0, 1, 2, 3, 4, 5]
  rews = [10, 20, 30, 40, 50]
  ```
  """

  def __init__(self, episode_length=5):
    assert episode_length >= 1
    self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=())
    self.action_space = gym.spaces.Box(low=0, high=np.inf, shape=())
    self.episode_length = episode_length
    self.timestep = None

  def reset(self):
    t, self.timestep = 0, 1
    return t

  def step(self, action):
    if self.timestep is None:
      raise RuntimeError("Need to reset before first step().")
    if self.timestep > self.episode_length:
      raise RuntimeError("Episode is over. Need to step().")
    if np.array(action) not in self.action_space:
      raise ValueError(f"Invalid action {action}")

    t, self.timestep = self.timestep, self.timestep + 1
    done = (t == self.episode_length)
    return t, t, done, {}


def _make_buffering_venv(error_on_premature_reset: bool,
                         ) -> BufferingWrapper:
  venv = DummyVecEnv([_CountingEnv] * 2)
  venv = BufferingWrapper(venv, error_on_premature_reset)
  venv.reset()
  return venv


def _assert_equal_scrambled_vectors(a: np.ndarray, b: np.ndarray) -> bool:
  """Returns True if `a` and `b` are identical up to sorting."""
  assert a.shape == b.shape
  assert a.ndim == 1
  np.testing.assert_equal(np.sort(a), np.sort(b))


def _join_transitions(trans_list: Sequence[rollout.Transitions],
                      ) -> rollout.Transitions:

  obs = np.concatenate(t.obs for t in trans_list)
  next_obs = np.concatenate(t.next_obs for t in trans_list)
  rews = np.concatenate(t.rews for t in trans_list)
  acts = np.concatenate(t.acts for t in trans_list)
  dones = np.concatenate(t.dones for t in trans_list)
  return rollout.Transitions(
    obs=obs, next_obs=next_obs, rews=rews, acts=acts, dones=dones)


def test_pop(episode_lengths: Sequence[int] = (5, 6),
             n_steps: int = 10,
             extra_pop_timesteps: Sequence[int] = ()):
  """Check pop_transitions() results for BufferWrapper.

  To make things easier to test, we use _CountingEnv where the observation
  is simply the episode timestep. The reward is 10x the timestep. Our action
  is 2.1x the timestep. There is an confusing offset for the observation because
  it has timestep 0 (due to reset()) and the other quantities don't, so here is
  an example of environment outputs and associated actions:

  ```
  episode_length = 5
  obs = [0, 1, 2, 3, 4, 5]  (len=6)
  acts = [0, 2.1, 4.2, ..., 8.4]  (len=5)
  rews = [10, ..., 50]  (len=5)
  ```

  Converted to `Transition`-format, this looks like:
  ```
  episode_length = 5
  obs = [0, 1, 2, 3, 4, 5]  (len=5)
  next_obs = [1, 2, 3, 4, 5]  (len=5)
  acts = [0, 2.1, 4.2, ..., 8.4]  (len=5)
  rews = [10, ..., 50]  (len=5)
  ```

  Args:
    episode_lengths: The number of timesteps before episode end in each dummy
      environment.
    n_steps: Number of times to call `step()` on the dummy environment.
    extra_pop_timesteps: By default, we only call `pop_*()` after `n_steps`
      calls to `step()`. For every unique positive `x` in `extra_pop_timesteps`,
      we also call `pop_*()` after the `x`th call to `step()`. All popped
      samples are concatenated before validating results at the end of this
      test case.
  """
  if not n_steps >= 1:
    raise ValueError(n_steps)
  for t in extra_pop_timesteps:
    if not 1 <= t <= n_steps:
      raise ValueError(t)

  venv = DummyVecEnv([lambda: _CountingEnv(episode_length=n)
                      for n in episode_lengths])
  venv_buffer = BufferingWrapper(venv)

  # To test `pop_transitions`, we will check that every obs, act, and rew
  # returned by `.reset()` and `.step()` is also returned by one of the
  # calls to `pop_transitions()`.
  transitions_list = []

  # Initial observation (only matters for pop_transitions()).
  obs = venv_buffer.reset()
  np.testing.assert_array_equal(obs, [0] * venv.num_envs)

  for t in range(1, n_steps + 1):
    acts = obs * 2.1
    venv_buffer.step_async(acts)
    obs, *_ = venv_buffer.step_wait()

    if t in extra_pop_timesteps:
      transitions_list.extend(venv_buffer.pop_transitions())

  transitions_list.extend(venv_buffer.pop_transitions())

  # Build expected transitions
  expect_obs = []
  for ep_len in episode_lengths:
    n_complete, remainder = divmod(n_steps, ep_len)
    expect_obs.extend([np.arange(ep_len)] * n_complete)
    expect_obs.append([np.arange(remainder)])

  expect_obs = np.concatenate(expect_obs)
  expect_next_obs = expect_obs + 1
  expect_acts = expect_obs * 2.1
  expect_rews = expect_next_obs * 10

  # Check `pop_transitions()`
  trans = _join_transitions(transitions_list)
  _assert_equal_scrambled_vectors(trans.obs, expect_obs)
  _assert_equal_scrambled_vectors(trans.next_obs, expect_next_obs)
  _assert_equal_scrambled_vectors(trans.acts, expect_acts)
  _assert_equal_scrambled_vectors(trans.rews, expect_rews)


def test_reset_error():
  # Resetting before a `step()` is okay.
  for flag in [True, False]:
    venv = _make_buffering_venv(flag)
    for _ in range(10):
      venv.reset()

  # Resetting after a `step()` is not okay if error flag is True.
  venv = _make_buffering_venv(True)
  venv.step([0, 0])
  with pytest.raises(RuntimeError, match="before samples were accessed"):
    venv.reset()

  # Same as previous case, but insert a `pop_transitions()` in between.
  venv = _make_buffering_venv(True)
  venv.step([0, 0])
  venv.pop_transitions()
  venv.step([0, 0])
  with pytest.raises(RuntimeError, match="before samples were accessed"):
    venv.reset()

  # Resetting after a `step()` is ok if error flag is False.
  venv = _make_buffering_venv(False)
  venv.step([0, 0])
  venv.reset()

  # Resetting after a `step()` is ok if transitions are first collected.
  for flag in [True, False]:
    venv = _make_buffering_venv(flag)
    venv.step([0, 0])
    venv.pop_transitions()
    venv.reset()


def test_mixed_pop_error():
  venv = _make_buffering_venv(True)
  venv.step([0, 0])
  venv.pop_transitions()
  venv.step([0, 0])
  with pytest.raises(RuntimeError, match=r".* pop types .*"):
    venv.pop_trajectories()

  venv = _make_buffering_venv(True)
  venv.step([0, 0])
  venv.pop_trajectories()
  venv.step([0, 0])
  with pytest.raises(RuntimeError, match=r".* pop types .*"):
    venv.pop_transitions()


def test_transitions_empty_after_pop():
  venv = _make_buffering_venv(True)
  venv.step([0, 0])
  venv.pop_transitions()
  result = venv.pop_transitions()
  assert len(result.obs) == 0
  assert len(result.rews) == 0
  assert len(result.acts) == 0
  assert len(result.rews) == 0
  assert len(result.dones) == 0


def test_trajectories_empty_after_pop():
  venv = _make_buffering_venv(True)
  venv.step([0, 0])
  venv.pop_trajectories()
  result = venv.pop_trajectories()
  assert len(result) == 0
