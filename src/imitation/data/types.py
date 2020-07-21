"""Types and helper methods for transitions and trajectories."""

import dataclasses
import os
import pickle
from typing import Optional, Sequence

import numpy as np
import tensorflow as tf

from imitation.data import old_types


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """A trajectory, e.g. a one episode rollout from an expert policy."""

    obs: np.ndarray
    """Observations, shape (trajectory_len + 1, ) + observation_shape."""

    acts: np.ndarray
    """Actions, shape (trajectory_len, ) + action_shape."""

    infos: Optional[np.ndarray]
    """An array of info dicts, length trajectory_len."""

    def __len__(self):
        """Returns number of transitions, `trajectory_len` in attribute docstrings.

        This is equal to the number of actions, and is always positive.
        """
        return len(self.acts)

    def __post_init__(self):
        """Performs input validation: check shapes are as specified in docstring."""
        if len(self.obs) != len(self.acts) + 1:
            raise ValueError(
                "expected one more observations than actions: "
                f"{len(self.obs)} != {len(self.acts)} + 1"
            )
        if self.infos is not None and len(self.infos) != len(self.acts):
            raise ValueError(
                "infos when present must be present for each action: "
                f"{len(self.infos)} != {len(self.acts)}"
            )
        if len(self.acts) == 0:
            raise ValueError("Degenerate trajectory: must have at least one action.")


def _rews_validation(rews: np.ndarray, acts: np.ndarray):
    if rews.shape != (len(acts),):
        raise ValueError(
            "rewards must be 1D array, one entry for each action: "
            f"{rews.shape} != ({len(acts)},)"
        )
    if rews.dtype not in [np.float32, np.float64, np.float128]:
        raise ValueError("rewards dtype {self.rews.dtype} not a float")


@dataclasses.dataclass(frozen=True)
class TrajectoryWithRew(Trajectory):
    rews: np.ndarray
    """Reward, shape (trajectory_len, ). dtype float."""

    def __post_init__(self):
        """Performs input validation, including for rews."""
        super().__post_init__()
        _rews_validation(self.rews, self.acts)


@dataclasses.dataclass(frozen=True)
class TransitionsMinimal:
    """A batch of obs-act transitions.

    This class and its subclasses are usually instantiated via
    `imitation.data.rollout.flatten_trajectories`.
    """

    obs: np.ndarray
    """
    Previous observations. Shape: (batch_size, ) + observation_shape.

    The i'th observation `obs[i]` in this array is the observation seen
    by the agent when choosing action `acts[i]`. `obs[i]` is not required to
    be from the timestep preceding `obs[i+1]`.
    """

    acts: np.ndarray
    """Actions. Shape: (batch_size,) + action_shape."""

    infos: np.ndarray
    """Array of info dicts. Shape: (batch_size,)."""

    def __len__(self):
        """Returns number of transitions. Always positive."""
        return len(self.obs)

    def __post_init__(self):
        """Performs input validation: check shapes & dtypes match docstring.

        Also make array values read-only.
        """
        for val in vars(self).values():
            if isinstance(val, np.ndarray):
                val.setflags(write=False)

        if len(self.obs) != len(self.acts):
            raise ValueError(
                "obs and acts must have same number of timesteps: "
                f"{len(self.obs)} != {len(self.acts)}"
            )
        if len(self.obs) == 0:
            raise ValueError("Must have non-zero number of observations.")

        if self.infos is not None and len(self.infos) != len(self.obs):
            raise ValueError(
                "obs and infos must have same number of timesteps: "
                f"{len(self.obs)} != {len(self.infos)}"
            )


@dataclasses.dataclass(frozen=True)
class Transitions(TransitionsMinimal):
    """A batch of obs-act-obs-done transitions."""

    next_obs: np.ndarray
    """New observation. Shape: (batch_size, ) + observation_shape.

    The i'th observation `next_obs[i]` in this array is the observation
    after the agent has taken action `acts[i]`.

    Invariants:
        * `next_obs.dtype == obs.dtype`
        * `len(next_obs) == len(obs)`
    """

    dones: np.ndarray
    """
    Boolean array indicating episode termination. Shape: (batch_size, ).

    `done[i]` is true iff `next_obs[i]` the last observation of an episode.
    """

    def __post_init__(self):
        """Performs input validation: check shapes & dtypes match docstring."""
        super().__post_init__()
        if self.obs.shape != self.next_obs.shape:
            raise ValueError(
                "obs and next_obs must have same shape: "
                f"{self.obs.shape} != {self.next_obs.shape}"
            )
        if self.obs.dtype != self.next_obs.dtype:
            raise ValueError(
                "obs and next_obs must have the same dtype: "
                f"{self.obs.dtype} != {self.next_obs.dtype}"
            )
        if self.dones.shape != (len(self.acts),):
            raise ValueError(
                "dones must be 1D array, one entry for each timestep: "
                f"{self.dones.shape} != ({len(self.acts)},)"
            )
        if self.dones.dtype != np.bool:
            raise ValueError(f"dones must be boolean, not {self.dones.dtype}")


@dataclasses.dataclass(frozen=True)
class TransitionsWithRew(Transitions):
    """A batch of obs-act-obs-rew-done transitions."""

    rews: np.ndarray
    """
    Reward. Shape: (batch_size, ). dtype float.

    The reward `rew[i]` at the i'th timestep is received after the
    agent has taken action `acts[i]`.
    """

    def __post_init__(self):
        """Performs input validation, including for rews."""
        super().__post_init__()
        _rews_validation(self.rews, self.acts)


def load(path: str) -> Sequence[TrajectoryWithRew]:
    """Loads a sequence of trajectories saved by `save()` from `path`."""
    # TODO(adam): remove backwards compatibility logic eventually (2021?)
    import sys

    try:
        assert "imitation.util.rollout" not in sys.modules
        sys.modules["imitation.util.rollout"] = old_types
        with open(path, "rb") as f:
            trajectories = pickle.load(f)
    finally:
        del sys.modules["imitation.util.rollout"]

    if len(trajectories) > 0:
        if isinstance(trajectories[0], old_types.Trajectory):
            trajectories = [
                TrajectoryWithRew(**traj._asdict()) for traj in trajectories
            ]

    return trajectories


def save(path: str, trajectories: Sequence[TrajectoryWithRew]) -> None:
    """Save a sequence of Trajectories to disk.

    Args:
        path: Trajectories are saved to this path.
        trajectories: The trajectories to save.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(trajectories, f)
    # Ensure atomic write
    os.replace(path + ".tmp", path)
    tf.logging.info("Dumped demonstrations to {}.".format(path))
