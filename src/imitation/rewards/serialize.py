"""Load serialized reward functions of different types."""

import contextlib
from typing import Callable, ContextManager, Iterator

import numpy as np
from stable_baselines.common.vec_env import VecEnv

from imitation.rewards import common, discrim_net, reward_net
from imitation.util import networks, registry, util

RewardFnLoaderFn = Callable[[str, VecEnv], ContextManager[common.RewardFn]]

reward_registry: registry.Registry[RewardFnLoaderFn] = registry.Registry()


@registry.sess_context
def _load_discrim_net(path: str, venv: VecEnv) -> common.RewardFn:
    """Load test reward output from discriminator."""
    del venv  # Unused.
    discriminator = discrim_net.DiscrimNet.load(path)
    # TODO(gleave): expose train reward as well? (hard due to action probs?)
    return discriminator.reward_test


def _load_reward_net_as_fn(shaped: bool) -> RewardFnLoaderFn:
    @contextlib.contextmanager
    def loader(path: str, venv: VecEnv) -> Iterator[common.RewardFn]:
        """Load train (shaped) or test (not shaped) reward from path."""
        del venv  # Unused.
        with networks.make_session() as (graph, sess):
            net = reward_net.RewardNet.load(path)
            reward = net.reward_output_train if shaped else net.reward_output_test

            def rew_fn(
                obs: np.ndarray,
                act: np.ndarray,
                next_obs: np.ndarray,
                dones: np.ndarray,
            ) -> np.ndarray:
                fd = {
                    net.obs_ph: obs,
                    net.act_ph: act,
                    net.next_obs_ph: next_obs,
                    net.done_ph: dones,
                }
                rew = sess.run(reward, feed_dict=fd)
                assert rew.shape == (len(obs),)
                return rew

            yield rew_fn

    return loader


def load_zero(path: str, venv: VecEnv) -> common.RewardFn:
    del path, venv

    def f(
        obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray, dones: np.ndarray
    ) -> np.ndarray:
        del act, next_obs, dones  # Unused.
        return np.zeros(obs.shape[0])

    return f


reward_registry.register(key="DiscrimNet", value=_load_discrim_net)
reward_registry.register(
    key="RewardNet_shaped", value=_load_reward_net_as_fn(shaped=True)
)
reward_registry.register(
    key="RewardNet_unshaped", value=_load_reward_net_as_fn(shaped=False)
)
reward_registry.register(key="zero", value=registry.dummy_context(load_zero))


@util.docstring_parameter(reward_types=", ".join(reward_registry.keys()))
def load_reward(
    reward_type: str, reward_path: str, venv: VecEnv
) -> ContextManager[common.RewardFn]:
    """Load serialized policy.

    Args:
      reward_type: A key in `reward_registry`. Valid types
          include {reward_types}.
      reward_path: A path specifying the reward.
      venv: An environment that the policy is to be used with.
    """
    reward_loader = reward_registry.get(reward_type)
    return reward_loader(reward_path, venv)
