"""Serialization utilities for trajectories."""
import logging
import os
import warnings
from typing import Mapping, Sequence, cast

import datasets
import huggingface_sb3 as hfsb3
import numpy as np

from imitation.data import huggingface_utils
from imitation.data.types import AnyPath, Trajectory, TrajectoryWithRew
from imitation.util import util


def save(path: AnyPath, trajectories: Sequence[Trajectory]) -> None:
    """Save a sequence of Trajectories to disk using HuggingFace's datasets library.

    Args:
        path: Trajectories are saved to this path.
        trajectories: The trajectories to save.
    """
    p = util.parse_path(path)
    huggingface_utils.trajectories_to_dataset(trajectories).save_to_disk(p)
    logging.info(f"Dumped demonstrations to {p}.")


def load(path: AnyPath) -> Sequence[Trajectory]:
    """Loads a sequence of trajectories saved by `save()` from `path`."""
    # Interestingly, np.load will just silently load a normal pickle file when you
    # set `allow_pickle=True`. So this call should succeed for both the new compressed
    # .npz format and the old pickle based format. To tell the difference, we need to
    # look at the type of the resulting object. If it's the new compressed format,
    # it should be a Mapping that we need to decode, whereas if it's the old format,
    # it's just the sequence of trajectories, and we can return it directly.

    if os.path.isdir(path):  # huggingface datasets format
        dataset = datasets.load_from_disk(str(path))
        if not isinstance(dataset, datasets.Dataset):  # pragma: no cover
            raise ValueError(
                f"Expected to load a `datasets.Dataset` but got {type(dataset)}",
            )

        return huggingface_utils.TrajectoryDatasetSequence(dataset)

    data = np.load(path, allow_pickle=True)  # works for both .npz and .pkl

    if isinstance(data, Sequence):  # pickle format
        warnings.warn("Loading old pickle version of Trajectories", DeprecationWarning)
        return data
    if isinstance(data, Mapping):  # .npz format
        warnings.warn("Loading old npz version of Trajectories", DeprecationWarning)
        num_trajs = len(data["indices"])
        fields = [
            # Account for the extra obs in each trajectory
            np.split(data["obs"], data["indices"] + np.arange(num_trajs) + 1),
            np.split(data["acts"], data["indices"]),
            np.split(data["infos"], data["indices"]),
            data["terminal"],
        ]
        if "rews" in data:
            fields = [
                *fields,
                np.split(data["rews"], data["indices"]),
            ]
            return [TrajectoryWithRew(*args) for args in zip(*fields)]
        else:
            return [Trajectory(*args) for args in zip(*fields)]  # pragma: no cover
    else:  # pragma: no cover
        raise ValueError(
            f"Expected either an .npz file or a pickled sequence of trajectories; "
            f"got a pickled object of type {type(data).__name__}",
        )


def load_with_rewards(path: AnyPath) -> Sequence[TrajectoryWithRew]:
    """Loads a sequence of trajectories with rewards from a file."""
    data = load(path)

    mismatched_types = [
        type(traj) for traj in data if not isinstance(traj, TrajectoryWithRew)
    ]
    if mismatched_types:
        raise ValueError(
            f"Expected all trajectories to be of type `TrajectoryWithRew`, "
            f"but found {mismatched_types[0].__name__}",
        )

    return cast(Sequence[TrajectoryWithRew], data)


def load_rollouts_from_huggingface(
    algo_name: str,
    env_name: str,
    organization: str = "HumanCompatibleAI",
) -> str:
    model_name = hfsb3.ModelName(algo_name, hfsb3.EnvironmentName(env_name))
    repo_id = hfsb3.ModelRepoId(organization, model_name)
    filename = hfsb3.load_from_hub(repo_id, "rollouts.npz")
    return filename
