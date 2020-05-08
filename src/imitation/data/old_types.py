from typing import List, NamedTuple, Optional

import numpy as np


class Trajectory(NamedTuple):
    """A trajectory, e.g. a one episode rollout from an expert policy."""

    acts: np.ndarray
    """Actions, shape (trajectory_len, ) + action_shape."""

    obs: np.ndarray
    """Observations, shape (trajectory_len + 1, ) + observation_shape."""

    rews: np.ndarray
    """Reward, shape (trajectory_len, )."""

    infos: Optional[List[dict]]
    """A list of info dicts, length trajectory_len."""
