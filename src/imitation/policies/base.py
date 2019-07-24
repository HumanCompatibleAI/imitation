"""Custom policy classes and convenience methods."""

import abc

import gym
import numpy as np
from stable_baselines.common import BaseRLModel
from stable_baselines.common.policies import BasePolicy


class HardCodedPolicy(BasePolicy):
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