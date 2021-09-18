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


def _load_reward_net_as_fn(shaped: bool) -> RewardFnLoaderFn:
    def loader(path: str, venv: VecEnv) -> common.RewardFn:
        """Load train (shaped) or test (not shaped) reward from path."""
        del venv  # Unused.
        net = th.load(str(path))
        if not shaped and hasattr(net, "base"):
            # If the "base" attribute exists, we are dealing with a ShapedRewardNet
            # and will disable the potential shaping (if shaped is False).
            # If no "base" attribute exists, we seem to use an unshaped RewardNet
            # anyway, so we just use its predict() method directly.
            reward = net.base.predict
        else:
            reward = net.predict

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
        obs: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        del act, next_obs, dones  # Unused.
        return np.zeros(obs.shape[0])

    return f


# TODO(adam): I think we can get rid of this and have just one RewardNet.
reward_registry.register(
    key="RewardNet_shaped",
    value=_load_reward_net_as_fn(shaped=True),
)
reward_registry.register(
    key="RewardNet_unshaped",
    value=_load_reward_net_as_fn(shaped=False),
)
reward_registry.register(key="zero", value=load_zero)


@util.docstring_parameter(reward_types=", ".join(reward_registry.keys()))
def load_reward(reward_type: str, reward_path: str, venv: VecEnv) -> common.RewardFn:
    """Load serialized reward.

    Args:
        reward_type: A key in `reward_registry`. Valid types
            include {reward_types}.
        reward_path: A path specifying the reward.
        venv: An environment that the policy is to be used with.

    Returns:
        The deserialized reward.
    """
    reward_loader = reward_registry.get(reward_type)
    return reward_loader(reward_path, venv)
