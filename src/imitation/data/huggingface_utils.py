"""Helpers to convert between Trajectories and HuggingFace's datasets library."""
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

        def numpy_transform(batch):
            return {key: np.asarray(val) for key, val in batch.items()}

        # TODO: this is just a temporary workaround for
        #  https://github.com/huggingface/datasets/issues/5517
        #  switch to .with_format("numpy") once it's fixed
        self._dataset = dataset.with_transform(numpy_transform)
        self._trajectory_class = (
            types.TrajectoryWithRew if "rews" in dataset.features else types.Trajectory
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Note: we could use self._dataset[idx] here and then convert the result of
            #   that to a series of trajectories, but if we do that, we run into trouble
            #   with the custom numpy transform that we apply in the constructor.
            #   The transform is applied to the whole slice, which might contain
            #   trajectories of different lengths which is not supported by numpy.
            return [self[i] for i in range(*idx.indices(len(self)))]
        else:
            # Extract the trajectory kwargs from the dataset
            kwargs = self._dataset[idx]

            # Ensure that the infos are decoded lazily using jsonpickle
            kwargs["infos"] = _LazyDecodedList(kwargs["infos"])

            return self._trajectory_class(**kwargs)

    @property
    def dataset(self):
        """Return the underlying HF dataset."""
        # Note: since we apply the custom numpy transform in the constructor, we remove
        #   it again before returning the dataset. This ensures that the dataset is
        #   returned in the original format and can be saved to disk
        #   (the custom transform can not be saved to disk since it is not pickleable).
        return self._dataset.with_transform(None)


class _LazyDecodedList(Sequence[Any]):
    """A wrapper to lazily decode a list of jsonpickled strings.

    Decoded results are cached to avoid decoding the same string multiple times.

    This is used to decode the infos of a trajectory only when they are accessed.
    """

    def __init__(self, encoded_list: Sequence[str]):
        self._encoded_list = encoded_list
        self._decoded_cache: Dict[int, Any] = {}

    def __len__(self):
        return len(self._encoded_list)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        else:
            if idx not in self._decoded_cache:
                self._decoded_cache[idx] = jsonpickle.decode(self._encoded_list[idx])
            return self._decoded_cache[idx]


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
    if any(isinstance(traj.obs, types.DictObs) for traj in trajectories):
        raise ValueError("DictObs are not currently supported")

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
