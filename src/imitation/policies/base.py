"""Custom policy classes and convenience methods."""

import abc

import gym
import numpy as np
from stable_baselines.a2c.utils import conv, conv_to_fc, linear
from stable_baselines.common import BaseRLModel
from stable_baselines.common.policies import BasePolicy, FeedForwardPolicy
import tensorflow as tf


def mnist_cnn(scaled_images, **kwargs):
  """
  Tweakeable CNN.
  :param scaled_images: (TensorFlow Tensor) Image input placeholder
  :param kwargs: (dict) Extra keywords parameters
  :return: (TensorFlow Tensor) The CNN output layer
  """
  activ = tf.nn.relu
  layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3,
                  stride=1, **kwargs))
  layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=2,
                  **kwargs))
  layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1,
                  **kwargs))
  layer_3 = conv_to_fc(layer_3)
  return activ(linear(layer_3, 'fc1', n_hidden=128))


class HardCodedPolicy(BasePolicy, abc.ABC):
  """Abstract class for hard-coded (non-trainable) policies."""
  def __init__(self, ob_space: gym.Space, ac_space: gym.Space):
    self.ob_space = ob_space
    self.ac_space = ac_space

  def step(self, obs, state=None, mask=None, deterministic=False):
    actions = []
    for ob in obs:
      assert self.ob_space.contains(ob)
      actions.append(self._choose_action(obs))
    return actions, None, None, None

  @abc.abstractmethod
  def _choose_action(self, obs):
    """Chooses an action, optionally based on observation obs."""

  def proba_step(self, obs, state=None, mask=None):
    raise NotImplementedError()


class RandomPolicy(HardCodedPolicy):
  """Returns random actions."""
  def __init__(self, ob_space: gym.Space, ac_space: gym.Space):
    super().__init__(ob_space, ac_space)

  def _choose_action(self, obs):
    return self.ac_space.sample()


class ZeroPolicy(HardCodedPolicy):
  """Returns constant zero action."""
  def __init__(self, ob_space: gym.Space, ac_space: gym.Space):
    super().__init__(ob_space, ac_space)

  def _choose_action(self, obs):
    return np.zeros(self.ac_space.shape, dtype=self.ac_space.dtype)


class FeedForward32Policy(FeedForwardPolicy):
  """A feed forward policy network with two hidden layers of 32 units.

  This matches the IRL policies in the original AIRL paper.

  Note: This differs from stable_baselines MlpPolicy in two ways: by having
  32 rather than 64 units, and by having policy and value networks share weights
  except at the final layer.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs,
                     net_arch=[32, 32], feature_extraction="mlp")


def get_action_policy(policy, observation, deterministic=True):
  """Gets an action from a Stable Baselines policy after some processing.

  Specifically, clips actions to the action space associated with `policy` and
  automatically accounts for vectorized environments inputs.

  This code was adapted from Stable Baselines' `BaseRLModel.predict()`.

  Args:
    policy (stable_baselines.common.policies.BasePolicy): The policy.
    observation (np.ndarray): The input to the policy network. Can either
      be a single input with shape `policy.ob_space.shape` or a vectorized
      input with shape `(n_batch,) + policy.ob_space.shape`.
    deterministic (bool): Whether or not to return deterministic actions
      (usually means argmax over policy's action distribution).

  Returns:
    action (np.ndarray): The action output of the policy network. If
        `observation` is not vectorized (has shape `policy.ob_space.shape`
        instead of shape `(n_batch,) + policy.ob_space.shape`) then
        `action` has shape `policy.ac_space.shape`.
        Otherwise, `action` has shape `(n_batch,) + policy.ac_space.shape`.
  """
  observation = np.array(observation)
  vectorized_env = BaseRLModel._is_vectorized_observation(observation,
                                                          policy.ob_space)

  observation = observation.reshape((-1, ) + policy.ob_space.shape)
  actions, _, states, _ = policy.step(observation, deterministic=deterministic)

  clipped_actions = actions
  if isinstance(policy.ac_space, gym.spaces.Box):
    clipped_actions = np.clip(actions, policy.ac_space.low,
                              policy.ac_space.high)

  if not vectorized_env:
    clipped_actions = clipped_actions[0]

  return clipped_actions, states


class MnistCnnPolicy(FeedForwardPolicy):
  """
  Policy object that implements actor critic, using a CNN (the Mnist CNN)
  :param sess: (TensorFlow session) The current TensorFlow session
  :param ob_space: (Gym Space) The observation space of the environment
  :param ac_space: (Gym Space) The action space of the environment
  :param n_env: (int) The number of environments to run
  :param n_steps: (int) The number of steps to run for each environment
  :param n_batch: (int) The number of batch to run (n_envs * n_steps)
  :param reuse: (bool) If the policy is reusable or not
  :param _kwargs: (dict) Extra keyword arguments for feature extraction
  """

  def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
               reuse=False, **_kwargs):
      super(MnistCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env,
                                           n_steps, n_batch, reuse,
                                           feature_extraction="cnn",
                                           cnn_extractor=mnist_cnn, **_kwargs)
