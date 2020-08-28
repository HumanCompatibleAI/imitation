"""Load serialized reward functions of different types."""

from typing import Callable

import numpy as np
import torch as th
from stable_baselines3.common.vec_env import VecEnv

from imitation.rewards import common
from imitation.util import registry, util

# TODO(sam): I suspect this whole file can be replaced with th.load calls. Try
# that refactoring once I have things running.

RewardFnLoaderFn = Callable[[str, VecEnv], common.RewardFn]

reward_registry: registry.Registry[RewardFnLoaderFn] = registry.Registry()


def _load_discrim_net(path: str, venv: VecEnv) -> common.RewardFn:
    """Load test reward output from discriminator."""
    del venv  # Unused.
    discriminator = th.load(path)
    # TODO(gleave): expose train reward as well? (hard due to action probs?)
    return discriminator.predict_reward_test


def _load_reward_net_as_fn(shaped: bool) -> RewardFnLoaderFn:
    def loader(path: str, venv: VecEnv) -> common.RewardFn:
        """Load train (shaped) or test (not shaped) reward from path."""
        del venv  # Unused.
        net = th.load(str(path))
        reward = net.predict_reward_train if shaped else net.predict_reward_test

        def rew_fn(
            obs: np.ndarray,
            act: np.ndarray,
            next_obs: np.ndarray,
            dones: np.ndarray,
        ) -> np.ndarray:
            rew = reward(obs, act, next_obs, dones)
            assert rew.shape == (len(obs),)
            return rew

        return rew_fn

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
reward_registry.register(key="zero", value=load_zero)


@util.docstring_parameter(reward_types=", ".join(reward_registry.keys()))
def load_reward(reward_type: str, reward_path: str, venv: VecEnv) -> common.RewardFn:
    """Load serialized policy.

    Args:
      reward_type: A key in `reward_registry`. Valid types
          include {reward_types}.
      reward_path: A path specifying the reward.
      venv: An environment that the policy is to be used with.
    """
    reward_loader = reward_registry.get(reward_type)
    return reward_loader(reward_path, venv)
