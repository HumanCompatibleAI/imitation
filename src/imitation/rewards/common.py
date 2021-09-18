"""Utilities and definitions shared by reward-related code."""

from typing import Callable

import numpy as np
from stable_baselines3.common import vec_env

RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def build_norm_reward_fn(
    *,
    reward_fn: RewardFn,
    vec_normalize: vec_env.VecNormalize,
    norm_reward: bool = True,
) -> RewardFn:
    """Wraps `reward_fn` to automatically normalize input.

    Args:
        reward_fn: The reward function that normalized inputs are evaluated on.
        vec_normalize: Instance of VecNormalize used to normalize inputs and
            rewards.
        norm_reward: If True, then also normalize reward before returning.

    Returns:
        A reward function that normalizes the inputs using `vec_normalize`,
        calls `reward_fn` and if `norm_reward` then normalizes the reward.
    """

    def inner(
        obs: np.ndarray,
        acts: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Normalizes `obs` and `next_obs` and computes reward from `reward_fn`.

        Args:
            obs: Observations before transition.
            acts: Actions.
            next_obs: Observations after transition.
            dones: Is the transition into terminal state at end of episode?

        Returns:
            The reward, normalized if `norm_reward` is true.
        """
        norm_obs = vec_normalize.normalize_obs(obs)
        norm_next_obs = vec_normalize.normalize_obs(next_obs)
        rew = reward_fn(norm_obs, acts, norm_next_obs, dones)
        if norm_reward:
            rew = vec_normalize.normalize_reward(rew)
        return rew

    return inner
