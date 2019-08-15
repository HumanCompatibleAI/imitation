"""Load serialized policies of different types."""

from typing import Callable

import numpy as np
from stable_baselines.common.vec_env import VecEnv

from imitation.rewards import discrim_net, reward_net
from imitation.util import registry, util

RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
RewardLoaderFn = Callable[[str, VecEnv], RewardFn]
RewardNetLoaderFn = Callable[[str, VecEnv], reward_net.RewardNet]

reward_net_registry: registry.Registry[RewardNetLoaderFn] = registry.Registry()
reward_registry: registry.Registry[RewardLoaderFn] = registry.Registry()


def _add_reward_net_loaders(classes):
  for name, cls in classes.items():
    loader = registry.build_loader_fn_require_path(cls.load)
    reward_net_registry.register(key=name, value=loader)


_add_reward_net_loaders({
    "BasicRewardNet": reward_net.BasicRewardNet,
    "BasicShapedRewardNet": reward_net.BasicShapedRewardNet,
})


def _load_discrim_net(cls):
  def f(path: str, venv: VecEnv):
    discrim_net = cls.load(path)
    # TODO(gleave): expose train reward as well? (hard due to action probs?)
    return discrim_net.reward_test
  return f


def _add_discrim_net_loaders(classes):
  for name, cls in classes.items():
    reward_registry.register(key=name, value=_load_discrim_net(cls))


_add_discrim_net_loaders({
    "DiscrimNetAIRL": discrim_net.DiscrimNetAIRL,
    "DiscrimNetGAIL": discrim_net.DiscrimNetGAIL,
})


def load_reward_net_as_fn(path: str, env: VecEnv) -> RewardFn:
  reward_type, shaped, reward_path = path.split(':')
  reward_net_loader = reward_net_registry.get(reward_type)
  shaped = bool(shaped)

  # TODO(adam): leaks session
  with util.make_session(close_on_exit=False) as sess:
    net = reward_net_loader(path, env)
    reward = net.reward_output_train if shaped else net.reward_output_test

    def f(old_obs: np.ndarray,
          act: np.ndarray,
          new_obs: np.ndarray) -> np.ndarray:
      fd = {
          net.old_obs_ph: old_obs,
          net.act_ph: act,
          net.new_obs_ph: new_obs,
      }
      rew = sess.run(reward, feed_dict=fd)
      assert rew.shape == (len(old_obs), )
      return rew

  return f


def load_zero(path: str, env: VecEnv) -> RewardFn:
  def f(old_obs: np.ndarray,
        act: np.ndarray,
        new_obs: np.ndarray) -> np.ndarray:
    return np.zeros(old_obs.shape[0])
  return f


reward_registry.register(key='zero', value=load_zero)
reward_registry.register(key="RewardNet", value=load_reward_net_as_fn)


def load_reward(reward_type: str, reward_path: str,
                venv: VecEnv) -> RewardFn:
  """Load serialized policy.

  Args:
    reward_type: A key in `reward_registry`, e.g. `RewardNet`.
    reward_path: A path specifying the reward.
    venv: An environment that the policy is to be used with.
  """
  reward_loader = reward_registry.get(reward_type)
  return reward_loader(reward_path, venv)
