"""Load serialized reward functions of different types."""

from typing import Callable

import numpy as np
from stable_baselines.common.vec_env import VecEnv

from imitation.rewards import discrim_net, reward_net
from imitation.util import registry, util
from imitation.util.reward_wrapper import RewardFn

RewardLoaderFn = Callable[[str, VecEnv], RewardFn]
RewardNetLoaderFn = Callable[[str, VecEnv], reward_net.RewardNet]

reward_net_registry: registry.Registry[RewardNetLoaderFn] = registry.Registry()
reward_fn_registry: registry.Registry[RewardLoaderFn] = registry.Registry()


def _add_reward_net_loaders(classes):
  for name, cls in classes.items():
    loader = registry.build_loader_fn_require_path(cls.load)
    reward_net_registry.register(key=name, value=loader)


REWARD_NETS = {
    "BasicRewardNet": reward_net.BasicRewardNet,
    "BasicShapedRewardNet": reward_net.BasicShapedRewardNet,
}
_add_reward_net_loaders(REWARD_NETS)


def _load_discrim_net(cls):
  def f(path: str, venv: VecEnv):
    # TODO(adam): leaks session
    with util.make_session(close_on_exit=False) as (graph, sess):
      discrim_net = cls.load(path)
      # TODO(gleave): expose train reward as well? (hard due to action probs?)
      return discrim_net.reward_test
  return f


def _add_discrim_net_loaders(classes):
  for name, cls in classes.items():
    reward_fn_registry.register(key=name, value=_load_discrim_net(cls))


_add_discrim_net_loaders({
    "DiscrimNetAIRL": discrim_net.DiscrimNetAIRL,
    "DiscrimNetGAIL": discrim_net.DiscrimNetGAIL,
})


def _load_reward_net_as_fn(reward_type: str, shaped: bool) -> RewardLoaderFn:
  reward_net_loader = reward_net_registry.get(reward_type)

  def loader(path: str, venv: VecEnv) -> RewardFn:
    with util.make_session(close_on_exit=False) as (graph, sess):
      net = reward_net_loader(path, venv)
    reward = net.reward_output_train if shaped else net.reward_output_test

    def rew_fn(old_obs: np.ndarray,
               act: np.ndarray,
               new_obs: np.ndarray,
               steps: np.ndarray) -> np.ndarray:
      fd = {
          net.old_obs_ph: old_obs,
          net.act_ph: act,
          net.new_obs_ph: new_obs,
      }
      rew = sess.run(reward, feed_dict=fd)
      assert rew.shape == (len(old_obs), )
      return rew

    return rew_fn

  return loader


def _add_reward_net_as_fn_loaders(reward_nets):
  for k, net_cls in reward_nets.items():
    reward_fn_registry.register(key=f"{k}_shaped",
                                value=_load_reward_net_as_fn(k, True))
    reward_fn_registry.register(key=f"{k}_unshaped",
                                value=_load_reward_net_as_fn(k, False))


_add_reward_net_as_fn_loaders(REWARD_NETS)


def load_zero(path: str, venv: VecEnv) -> RewardFn:
  def f(old_obs: np.ndarray,
        act: np.ndarray,
        new_obs: np.ndarray,
        steps: np.ndarray) -> np.ndarray:
    return np.zeros(old_obs.shape[0])
  return f


reward_fn_registry.register(key='zero', value=load_zero)


@util.docstring_parameter(reward_types=", ".join(reward_fn_registry.keys()))
def load_reward(reward_type: str, reward_path: str,
                venv: VecEnv) -> RewardFn:
  """Load serialized policy.

  Args:
    reward_type: A key in `reward_registry`, e.g. `RewardNet`. Valid types
        include {reward_types}.
    reward_path: A path specifying the reward.
    venv: An environment that the policy is to be used with.
  """
  reward_loader = reward_fn_registry.get(reward_type)
  return reward_loader(reward_path, venv)
