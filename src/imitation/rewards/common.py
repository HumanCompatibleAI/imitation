"""Utilities and definitions shared by reward-related code."""

from typing import Callable

import numpy as np

RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
