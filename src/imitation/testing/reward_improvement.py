"""Utility functions used to check if rewards improved wrt to previous rewards."""
from typing import Iterable

import numpy as np
from scipy import stats


def is_significant_reward_improvement(
    old_rewards: Iterable[float],
    new_rewards: Iterable[float],
    p_value: float = 0.05,
) -> bool:
    """Checks if the new rewards are really better than the old rewards.

    Ensures that this is not just due to lucky sampling by a permutation test.

    Args:
        old_rewards: Iterable of "old" trajectory rewards (e.g. before training).
        new_rewards: Iterable of "new" trajectory rewards (e.g. after training).
        p_value: The maximum probability, that the old rewards are just as good as the
            new rewards, that we tolerate.

    Returns:
        True, if the new rewards are most probably better than the old rewards.
        For this, the probability, that the old rewards are just as good as the new
        rewards must be below `p_value`.

    >>> is_significant_reward_improvement((5, 6, 7, 4, 4), (7, 5, 9, 9, 12))
    True

    >>> is_significant_reward_improvement((5, 6, 7, 4, 4), (7, 5, 9, 7, 4))
    False

    >>> is_significant_reward_improvement((5, 6, 7, 4, 4), (7, 5, 9, 7, 4), p_value=0.3)
    True
    """
    permutation_test_result = stats.permutation_test(
        (old_rewards, new_rewards),
        statistic=lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
        vectorized=True,
        alternative="less",
    )

    return permutation_test_result.pvalue < p_value


def mean_reward_improved_by(
    old_rews: Iterable[float],
    new_rews: Iterable[float],
    min_improvement: float,
):
    """Checks if mean rewards improved wrt. to old rewards by a certain amount.

    Args:
        old_rews: Iterable of "old" trajectory rewards (e.g. before training).
        new_rews: Iterable of "new" trajectory rewards (e.g. after training).
        min_improvement: The minimum amount of improvement that we expect.

    Returns:
        True if the mean of the new rewards is larger than the mean of the old rewards
        by min_improvement.

    >>> mean_reward_improved_by([5, 8, 7], [8, 9, 10], 2)
    True

    >>> mean_reward_improved_by([5, 8, 7], [8, 9, 10], 5)
    False
    """
    improvement = np.mean(new_rews) - np.mean(old_rews)  # type: ignore[call-overload]
    return improvement >= min_improvement
