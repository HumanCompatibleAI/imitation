"""Serialization utilities for trajectories."""
import logging
import os
import pathlib
import warnings
from typing import Mapping, Optional, Sequence, cast

import datasets
import numpy as np

from imitation.data import huggingface_datasets_conversion as hfds
from imitation.data.types import AnyPath, Trajectory, TrajectoryWithRew


def save(path: AnyPath, trajectories: Sequence[Trajectory]):
    """Save a sequence of Trajectories to disk using HuggingFace's datasets library.

    The dataset has the following fields:
    * obs: The observations. Shape: (num_timesteps, obs_dim). dtype: float.
    * acts: The actions. Shape: (num_timesteps, act_dim). dtype: float.
    * infos: The infos. Shape: (num_timesteps, ). dtype: (jsonpickled) str.
    * terminal: The terminal flags. Shape: (num_timesteps, ). dtype: bool.
    * rews: The rewards. Shape: (num_timesteps, ). dtype: float. if applicable.

    Args:
        path: Trajectories are saved to this path.
        trajectories: The trajectories to save.
    """
    p = parse_path(path)
    hfds.trajectories_to_dataset(trajectories).save_to_disk(p)
    logging.info(f"Dumped demonstrations to {p}.")


def load(path: AnyPath) -> Sequence[Trajectory]:
    """Loads a sequence of trajectories saved by `save()` from `path`."""
    # Interestingly, np.load will just silently load a normal pickle file when you
    # set `allow_pickle=True`. So this call should succeed for both the new compressed
    # .npz format and the old pickle based format. To tell the difference we need to
    # look at the type of the resulting object. If it's the new compressed format,
    # it should be a Mapping that we need to decode, whereas if it's the old format
    # it's just the sequence of trajectories, and we can return it directly.

    if os.path.isdir(path):  # huggingface datasets format

        # TODO: this is just a temporary workaround for
        #  https://github.com/huggingface/datasets/issues/5517
        #  switch to .with_format("numpy") once it's fixed
        def numpy_transform(batch):
            return {key: np.asarray(val) for key, val in batch.items()}

        dataset = datasets.load_from_disk(str(path)).with_transform(numpy_transform)

        if not isinstance(dataset, datasets.Dataset):
            raise ValueError(
                f"Expected to load a `datasets.Dataset` but got "
                f"{type(dataset).__name__}",
            )

        return hfds.TrajectoryDatasetSequence(dataset)

    data = np.load(path, allow_pickle=True)
    if isinstance(data, Sequence):  # pickle format
        warnings.warn("Loading old version of Trajectory's", DeprecationWarning)
        return data
    elif isinstance(data, Mapping):  # .npz format
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
            return [Trajectory(*args) for args in zip(*fields)]
    else:
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


def parse_path(
    path: AnyPath,
    allow_relative: bool = True,
    base_directory: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    """Parse a path to a `pathlib.Path` object.

    All resulting paths are resolved, absolute paths. If `allow_relative` is True,
    then relative paths are allowed as input, and are resolved relative to the
    current working directory, or relative to `base_directory` if it is
    specified.

    Args:
        path: The path to parse. Can be a string, bytes, or `os.PathLike`.
        allow_relative: If True, then relative paths are allowed as input, and
            are resolved relative to the current working directory. If False,
            an error is raised if the path is not absolute.
        base_directory: If specified, then relative paths are resolved relative
            to this directory, instead of the current working directory.

    Returns:
        A `pathlib.Path` object.

    Raises:
        ValueError: If `allow_relative` is False and the path is not absolute.
        ValueError: If `base_directory` is specified and `allow_relative` is
            False.
    """
    if base_directory is not None and not allow_relative:
        raise ValueError(
            "If `base_directory` is specified, then `allow_relative` must be True.",
        )

    parsed_path: pathlib.Path
    if isinstance(path, pathlib.Path):
        parsed_path = path
    elif isinstance(path, str):
        parsed_path = pathlib.Path(path)
    elif isinstance(path, bytes):
        parsed_path = pathlib.Path(path.decode())
    else:
        parsed_path = pathlib.Path(str(path))

    if parsed_path.is_absolute():
        return parsed_path
    else:
        if allow_relative:
            base_directory = base_directory or pathlib.Path.cwd()
            # relative to current working directory
            return base_directory / parsed_path
        else:
            raise ValueError(f"Path {str(parsed_path)} is not absolute")


def parse_optional_path(
    path: Optional[AnyPath],
    allow_relative: bool = True,
    base_directory: Optional[pathlib.Path] = None,
) -> Optional[pathlib.Path]:
    """Parse an optional path to a `pathlib.Path` object.

    All resulting paths are resolved, absolute paths. If `allow_relative` is True,
    then relative paths are allowed as input, and are resolved relative to the
    current working directory, or relative to `base_directory` if it is
    specified.

    Args:
        path: The path to parse. Can be a string, bytes, or `os.PathLike`.
        allow_relative: If True, then relative paths are allowed as input, and
            are resolved relative to the current working directory. If False,
            an error is raised if the path is not absolute.
        base_directory: If specified, then relative paths are resolved relative
            to this directory, instead of the current working directory.

    Returns:
        A `pathlib.Path` object, or None if `path` is None.
    """
    if path is None:
        return None
    else:
        return parse_path(path, allow_relative, base_directory)
