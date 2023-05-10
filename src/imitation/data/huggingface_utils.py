"""Helpers to convert between Trajectories and HuggingFace's datasets library."""
import functools
from typing import Any, Dict, Iterable, Optional, Sequence, cast

import datasets
import jsonpickle
import numpy as np

from imitation.data import types


class TrajectoryDatasetSequence(Sequence[types.Trajectory]):
    """A wrapper to present an HF dataset as a sequence of trajectories.

    Converts the dataset to a sequence of trajectories on the fly.
    """

    def __init__(self, dataset: datasets.Dataset):
        """Construct a TrajectoryDatasetSequence."""
        # TODO: this is just a temporary workaround for
        #  https://github.com/huggingface/datasets/issues/5517
        #  switch to .with_format("numpy") once it's fixed
        def numpy_transform(batch):
            return {key: np.asarray(val) for key, val in batch.items()}

        self._dataset = dataset.with_transform(numpy_transform)
        self._trajectory_class = (
            types.TrajectoryWithRew if "rews" in dataset.features else types.Trajectory
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx):

        if isinstance(idx, slice):
            dataslice = self._dataset[idx]

            # Extract the trajectory kwargs from the dataset slice
            trajectory_kwargs = [
                {key: dataslice[key][i] for key in dataslice}
                for i in range(len(dataslice["obs"]))
            ]

            # Ensure that the infos are decoded lazily using jsonpickle
            for kwargs in trajectory_kwargs:
                kwargs["infos"] = _LazyDecodedList(kwargs["infos"])

            return [self._trajectory_class(**kwargs) for kwargs in trajectory_kwargs]
        else:
            # Extract the trajectory kwargs from the dataset
            kwargs = self._dataset[idx]

            # Ensure that the infos are decoded lazily using jsonpickle
            kwargs["infos"] = _LazyDecodedList(kwargs["infos"])

            return self._trajectory_class(**kwargs)

    @property
    def dataset(self):
        """Return the underlying HF dataset."""
        return self._dataset


class _LazyDecodedList(Sequence[Any]):
    """A wrapper to lazily decode a list of jsonpickled strings.

    Decoded results are cached to avoid decoding the same string multiple times.

    This is used to decode the infos of a trajectory only when they are accessed.
    """

    def __init__(self, encoded_list: Sequence[str]):
        self._encoded_list = encoded_list

    def __len__(self):
        return len(self._encoded_list)

    # arbitrary cache size just to put a limit on memory usage
    @functools.lru_cache(maxsize=100000)
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [jsonpickle.decode(info) for info in self._encoded_list[idx]]
        else:
            return jsonpickle.decode(self._encoded_list[idx])


def make_dict_from_trajectory(trajectory: types.Trajectory):
    """Convert a Trajectory to a dict.

    The dict has the following fields:
    * obs: The observations. Shape: (num_timesteps, obs_dim). dtype: float.
    * acts: The actions. Shape: (num_timesteps, act_dim). dtype: float.
    * infos: The infos. Shape: (num_timesteps, ). dtype: (jsonpickled) str.
    * terminal: The terminal flags. Shape: (num_timesteps, ). dtype: bool.
    * rews: The rewards. Shape: (num_timesteps, ). dtype: float. if applicable.

    Args:
        trajectory: The trajectory to convert.

    Returns:
        A dict representing the trajectory.
    """
    # Replace 'None' values for `infos`` with array of empty dicts
    infos = cast(
        Sequence[Dict[str, Any]],
        trajectory.infos if trajectory.infos is not None else [{}] * len(trajectory),
    )

    # Encode infos as jsonpickled strings
    encoded_infos = [jsonpickle.encode(info) for info in infos]

    trajectory_dict = dict(
        obs=trajectory.obs,
        acts=trajectory.acts,
        infos=encoded_infos,
        terminal=trajectory.terminal,
    )

    # Add rewards if applicable
    if isinstance(trajectory, types.TrajectoryWithRew):
        trajectory_dict["rews"] = trajectory.rews

    return trajectory_dict


def trajectories_to_dict(
    trajectories: Sequence[types.Trajectory],
) -> Dict[str, Sequence[Any]]:
    """Convert a sequence of trajectories to a dict.

    The dict has the following fields:

    * obs: The observations. Shape: (num_trajectories, num_timesteps, obs_dim).
    * acts: The actions. Shape: (num_trajectories, num_timesteps, act_dim).
    * infos: The infos. Shape: (num_trajectories, num_timesteps) as jsonpickled str.
    * terminal: The terminal flags. Shape: (num_trajectories, num_timesteps, ).
    * rews: The rewards. Shape: (num_trajectories, num_timesteps) if applicable.

    This dict can be used to construct a HuggingFace dataset.

    Args:
        trajectories: The trajectories to save.

    Raises:
        ValueError: If not all trajectories have the same type, i.e. some are
            `Trajectory` and others are `TrajectoryWithRew`.

    Returns:
        A dict representing the trajectories.
    """
    # Check that all trajectories have rewards or none have rewards
    has_reward = [isinstance(traj, types.TrajectoryWithRew) for traj in trajectories]
    all_trajectories_have_reward = all(has_reward)
    if not all_trajectories_have_reward and any(has_reward):
        raise ValueError("Some trajectories have rewards but not all")

    # Convert to dict
    trajectory_dict: Dict[str, Sequence[Any]] = dict(
        obs=[traj.obs for traj in trajectories],
        acts=[traj.acts for traj in trajectories],
        # Replace 'None' values for `infos`` with array of empty dicts
        infos=[
            traj.infos if traj.infos is not None else [{}] * len(traj)
            for traj in trajectories
        ],
        terminal=[traj.terminal for traj in trajectories],
    )

    # Encode infos as jsonpickled strings
    trajectory_dict["infos"] = [
        [jsonpickle.encode(info) for info in traj_infos]
        for traj_infos in cast(Iterable[Iterable[Dict]], trajectory_dict["infos"])
    ]

    # Add rewards if applicable
    if all_trajectories_have_reward:
        trajectory_dict["rews"] = [
            cast(types.TrajectoryWithRew, traj).rews for traj in trajectories
        ]
    return trajectory_dict


def trajectories_to_dataset(
    trajectories: Sequence[types.Trajectory],
    info: Optional[datasets.DatasetInfo] = None,
) -> datasets.Dataset:
    """Convert a sequence of trajectories to a HuggingFace dataset."""
    if isinstance(trajectories, TrajectoryDatasetSequence):
        return trajectories.dataset
    else:
        return datasets.Dataset.from_dict(trajectories_to_dict(trajectories), info=info)
