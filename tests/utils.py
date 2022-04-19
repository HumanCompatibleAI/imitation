"""Utility functions used to write tests."""
from typing import Iterable

import numpy as np
from scipy import stats


def rewards_improved(
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
            new reawards, that we tolerate.

    Returns:
        True, if the new rewards are most probably better than the old rewards.
        For this, the probability, that the old rewards are just as good as the new
        rewards must be below `p_value`.

    >>> rewards_improved((5, 6, 7, 4, 4), (7, 5, 9, 9, 12))
    True

    >>> rewards_improved((5, 6, 7, 4, 4), (7, 5, 9, 7, 4))
    False

    >>> rewards_improved((5, 6, 7, 4, 4), (7, 5, 9, 7, 4), p_value=0.3)
    True
    """
    permutation_test_result = stats.permutation_test(
        (old_rewards, new_rewards),
        statistic=lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
        vectorized=True,
        alternative="less",
    )

    return permutation_test_result.pvalue < p_value
