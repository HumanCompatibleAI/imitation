"""Load serialized reward functions of different types."""

import contextlib
from functools import partial
from typing import Callable, ContextManager, Iterator, Optional
from pathlib import Path
import pickle

import numpy as np
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common import BaseRLModel

from imitation.rewards import discrim_net, reward_net
from imitation.util import registry, util
from imitation.util.reward_wrapper import RewardFn

RewardFnLoaderFn = Callable[[str, VecEnv], ContextManager[RewardFn]]

reward_registry: registry.Registry[RewardFnLoaderFn] = registry.Registry()


@registry.sess_context
def _load_discrim_net_train(path: str,
                            venv: VecEnv,
                            gen_path: Optional[str] = None) -> RewardFn:
  """Load train reward output from discriminator."""
  del venv
  discriminator = discrim_net.DiscrimNet.load(path)
  if gen_path is None:
    gen_path = Path(path, "..", "gen_policy")

  gen_model_path = gen_path / "model.pkl"
  assert gen_model_path.exists()
  with open(gen_model_path, "rb") as f:
    gen_policy = pickle.load(f)  # type: BaseRLModel

  return partial(discriminator.reward_train,
                 gen_log_prob_fn=gen_policy.action_probability)


@registry.sess_context
def _load_discrim_net_test(path: str, venv: VecEnv) -> RewardFn:
  """Load test reward output from discriminator."""
  del venv
  discriminator = discrim_net.DiscrimNet.load(path)
  # TODO(gleave): expose train reward as well? (hard due to action probs?)
  return discriminator.reward_test


def _load_reward_net_as_fn(shaped: bool) -> RewardFnLoaderFn:
  @contextlib.contextmanager
  def loader(path: str,
             venv: VecEnv,
             ) -> Iterator[RewardFn]:
    """Load train (shaped) or test (not shaped) reward from path."""
    del venv
    with util.make_session() as (graph, sess):
      net = reward_net.RewardNet.load(path)
      reward = net.reward_output_train if shaped else net.reward_output_test

      def rew_fn(obs: np.ndarray,
                 act: np.ndarray,
                 next_obs: np.ndarray,
                 steps: np.ndarray) -> np.ndarray:
        del steps
        fd = {
            net.obs_ph: obs,
            net.act_ph: act,
            net.next_obs_ph: next_obs,
        }
        rew = sess.run(reward, feed_dict=fd)
        assert rew.shape == (len(obs),)
        return rew

      yield rew_fn

  return loader


def load_zero(path: str, venv: VecEnv) -> RewardFn:
  del path, venv

  def f(obs: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
        steps: np.ndarray) -> np.ndarray:
    del act, next_obs, steps
    return np.zeros(obs.shape[0])

  return f


reward_registry.register(key="DiscrimNet", value=_load_discrim_net_train)
reward_registry.register(key="DiscrimNet_train", value=_load_discrim_net_train)
reward_registry.register(key="DiscrimNet_test", value=_load_discrim_net_test)
reward_registry.register(key="RewardNet_shaped",
                         value=_load_reward_net_as_fn(shaped=True))
reward_registry.register(key="RewardNet_unshaped",
                         value=_load_reward_net_as_fn(shaped=False))
reward_registry.register(key='zero', value=registry.dummy_context(load_zero))


@util.docstring_parameter(reward_types=", ".join(reward_registry.keys()))
def load_reward(reward_type: str, reward_path: str,
                venv: VecEnv, **kwargs) -> ContextManager[RewardFn]:
  """Load serialized policy.

  Args:
    reward_type: A key in `reward_registry`. Valid types
        include {reward_types}.
    reward_path: A path specifying the reward.
    venv: An environment that the policy is to be used with.
    **kwargs: Optional additional arguments associated with a particular reward
      type. `DiscrimNet_train` takes an optional keyword argument `gen_path`,
      the path to the generator policy directory.
      (default is `{path}/../gen_policy`).
  """
  reward_loader = reward_registry.get(reward_type)
  return reward_loader(reward_path, venv, **kwargs)
