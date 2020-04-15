"""Tests for code that generates trajectory rollouts."""

import gym
import numpy as np
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

from imitation.policies import serialize
from imitation.policies.base import RandomPolicy
from imitation.util import rollout, util


class TerminalSentinelEnv(gym.Env):
  def __init__(self, max_acts):
    self.max_acts = max_acts
    self.current_step = 0
    self.action_space = gym.spaces.Discrete(1)
    self.observation_space = gym.spaces.Box(np.array([0]), np.array([1]))

  def reset(self):
    self.current_step = 0
    return np.array([0])

  def step(self, action):
    self.current_step += 1
    done = self.current_step >= self.max_acts
    observation = np.array([1 if done else 0])
    rew = 0.0
    return observation, rew, done, {}


def test_complete_trajectories():
  """Check that complete trajectories are returned by vecenv wrapper,
     including the terminal observation.
  """
  min_episodes = 13
  max_acts = 5
  num_envs = 4
  vec_env = DummyVecEnv([lambda: TerminalSentinelEnv(max_acts)] * num_envs)
  policy = RandomPolicy(vec_env.observation_space, vec_env.action_space)
  sample_until = rollout.min_episodes(min_episodes)
  trajectories = rollout.generate_trajectories(policy,
                                               vec_env,
                                               sample_until=sample_until)
  assert len(trajectories) >= min_episodes
  expected_obs = np.array([[0]] * max_acts + [[1]])
  for trajectory in trajectories:
    obs = trajectory.obs
    acts = trajectory.acts
    assert len(obs) == len(acts) + 1
    assert np.all(obs == expected_obs)


class ObsRewHalveWrapper(gym.Wrapper):
  """Simple wrapper that scales every reward and observation feature by 0.5."""

  def reset(self, **kwargs):
    obs = self.env.reset(**kwargs) / 2
    return obs

  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    return obs/2, rew/2, done, info


def test_rollout_stats():
  """Applying `ObsRewIncrementWrapper` halves the reward mean.

  `rollout_stats` should reflect this.
  """
  env = gym.make("CartPole-v1")
  env = Monitor(env, None)
  env = ObsRewHalveWrapper(env)
  venv = DummyVecEnv([lambda: env])

  with serialize.load_policy("zero", "UNUSED", venv) as policy:
    trajs = rollout.generate_trajectories(policy, venv,
                                          rollout.min_episodes(10))
  s = rollout.rollout_stats(trajs)

  np.testing.assert_allclose(s["return_mean"], s["monitor_return_mean"] / 2)
  np.testing.assert_allclose(s["return_std"], s["monitor_return_std"] / 2)
  np.testing.assert_allclose(s["return_min"], s["monitor_return_min"] / 2)
  np.testing.assert_allclose(s["return_max"], s["monitor_return_max"] / 2)


def test_unwrap_traj():
  """Check that unwrap_traj reverses `ObsRewIncrementWrapper`.

  Also check that unwrapping twice is a no-op."""
  env = gym.make("CartPole-v1")
  env = util.rollout.RolloutInfoWrapper(env)
  env = ObsRewHalveWrapper(env)
  venv = DummyVecEnv([lambda: env])

  with serialize.load_policy("zero", "UNUSED", venv) as policy:
    trajs = rollout.generate_trajectories(
      policy, venv, rollout.min_episodes(10))
  trajs_unwrapped = [rollout.unwrap_traj(t) for t in trajs]
  trajs_unwrapped_twice = [rollout.unwrap_traj(t) for t in trajs_unwrapped]

  for t, t_unwrapped in zip(trajs, trajs_unwrapped):
    np.testing.assert_allclose(t.acts, t_unwrapped.acts)
    np.testing.assert_allclose(t.obs, t_unwrapped.obs / 2)
    np.testing.assert_allclose(t.rews, t_unwrapped.rews / 2)

  for t1, t2 in zip(trajs_unwrapped, trajs_unwrapped_twice):
    np.testing.assert_equal(t1.acts, t2.acts)
    np.testing.assert_equal(t1.obs, t2.obs)
    np.testing.assert_equal(t1.rews, t2.rews)
