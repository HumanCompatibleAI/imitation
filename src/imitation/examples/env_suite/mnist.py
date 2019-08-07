"""Environment where AIRL should infer the correct reward function."""
import gym
import numpy as np
import tensorflow as tf

class MnistEnv(gym.Env):
  """A simple gridworld where:
      - observations are MNIST digits,
      - reward is the digit displayed
  """

  def __init__(self, map_size=2, nb_steps_per_ep=8):
    self.map_size = map_size
    self.nS = map_size ** 2
    self.nA = 4 #left, right, up, down
    self.nb_steps_per_ep = nb_steps_per_ep
    self.count_steps = 0

    # For gym
    self.action_space = gym.spaces.Discrete(self.nA)
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=(28,28))

    # Loading as many digits as cells in the gridworld
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    self.map = [
      [x_train[j] for j in range (map_size * i, map_size * (i + 1))]
      for i in range(map_size)]
    self.ground_truth = [
      [y_train[j] for j in range (map_size * i, map_size * (i + 1))]
      for i in range(map_size)]
    self.x, self.y = np.random.randint(map_size, size=2)

  def render(self):
    print(*self.ground_truth, sep='\n')

  def _move(self, direction):
    delta_x = [0, 0, -1, 1]
    delta_y = [-1, 1, 0, 0]
    self.x = (self.x + delta_x[direction]) % self.map_size
    self.y = (self.y + delta_y[direction]) % self.map_size

  def step(self, a):
    rew = self.ground_truth[self.x][self.y]
    self._move(a)
    obs = self.map[self.x][self.y]
    self.count_steps += 1
    return (obs, rew, self.count_steps >= self.nb_steps_per_ep, {})

  def reset(self):
    self.count_steps = 0
    self.x, self.y = np.random.randint(self.map_size, size=2)
    return self.map[self.x][self.y]