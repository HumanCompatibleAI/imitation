"""Tests for code that generates trajectory rollouts."""

import gym
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv

from imitation.policies.base import RandomPolicy
from imitation.util import rollout


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
    act = trajectory.act
    assert len(obs) == len(act) + 1
    assert np.all(obs == expected_obs)
