"""Environment where AIRL should infer the correct reward function."""

import gym
import numpy as np
import tensorflow as tf


class MnistEnv(gym.Env):
  """A simple gridworld where:
      - observations are MNIST digits,
      - actions are guesses in 0..9
      - reward = 1 if your action matches the true label, otherwise 0
  """

  def __init__(self):
    self.action_space = gym.spaces.Discrete(10)
    self.observation_space = gym.spaces.Box(low=0, high=1, shape=(28, 28, 1))
    (self.digits, self.labels), _ = tf.keras.datasets.mnist.load_data()
    self.digits = self.digits.reshape(self.digits.shape[0], 28, 28, 1) / 255
    self.idx = None

  def _sample_digit(self):
    self.idx = np.random.randint(len(self.labels))

  def step(self, a):
    assert self.idx is not None, "Cannot call env.step() before calling reset()"
    rew = float(a == self.labels[self.idx])
    return (np.zeros((28, 28, 1)), rew, True, {})

  def reset(self):
    self._sample_digit()
    return self.digits[self.idx]

  def seed(self, seed=None):
    np.random.seed(seed)
    return [seed]
