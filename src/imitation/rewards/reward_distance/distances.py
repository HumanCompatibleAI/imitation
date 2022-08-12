from typing import Callable, Union

import numpy as np

from imitation.rewards.reward_distance.distance_matrix import DistanceMatrix
from imitation.rewards.reward_distance.collections import RewardCollection


# pylint: disable=redefined-builtin
def compute_direct_distance(arr: np.ndarray, ord: Union[str, int, float] = 2) -> np.ndarray:
    """Computes the direct (norm) distance between reward arrays.

    Args:
        arr: 2D matrix of values. The correlation is computed between the rows of the array.
        ord: The type of norm to take (see documentation for `numpy.linalg.norm`).

    Returns:
        2D matrix of distance values between the rows of `arr`.
    """
    assert arr.ndim == 2
    assert not np.any(np.isnan(arr)), "NaN reward values found."
    differences = arr[:, None, :] - arr[None, :, :]
    norms = np.linalg.norm(differences, axis=2, ord=ord)
    return norms


def compute_pearson_correlation(arr: np.ndarray) -> np.ndarray:
    """Computes the correlation between the rows of the provided array.

    Args:
        arr: 2D matrix of values. The correlation is computed between the rows of the array.

    Returns:
        2D matrix of correlation values between the rows of `arr`.
    """
    assert arr.ndim == 2
    assert not np.any(np.isnan(arr)), "NaN reward values found."
    return np.corrcoef(arr)


def compute_pearson_distance(arr: np.ndarray) -> np.ndarray:
    """Computes the pearson distance between the rows of the provided array.

    Args:
        arr: 2D matrix of values. The distance is computed between the rows of the array.

    Returns:
        2D matrix of distance values between the rows of `arr`.
    """
    corr = compute_pearson_correlation(arr)
    return np.sqrt(0.5 * (1 - corr))


def compute_distance_between_reward_pairs(
    rewards: RewardCollection,
    distance_fn: Callable = compute_pearson_distance,
) -> DistanceMatrix:
    """Computes the distance between each pair of provided rewards.

    Args:
        rewards: Rewards to compute distances between.
        distance_fn: Distance function to use.

    Returns:
        A DistanceMatrix containing the distances.
    """
    labels = list(rewards.keys())
    reward_arrays = []
    for label in labels:
        reward_arrays.append(rewards[label].detach().cpu().numpy())
    reward_arrays = np.stack(reward_arrays)
    distances = distance_fn(reward_arrays)
    return DistanceMatrix(labels, distances)
