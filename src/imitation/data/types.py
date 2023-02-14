"""Types and helper methods for transitions and trajectories."""

import dataclasses
import logging
import os
import pathlib
import warnings
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import datasets
import numpy as np
import torch as th
from torch.utils import data as th_data

T = TypeVar("T")

AnyPath = Union[str, bytes, os.PathLike]


def dataclass_quick_asdict(obj) -> Dict[str, Any]:
    """Extract dataclass to items using `dataclasses.fields` + dict comprehension.

    This is a quick alternative to `dataclasses.asdict`, which expensively and
    undocumentedly deep-copies every numpy array value.
    See https://stackoverflow.com/a/52229565/1091722.

    Args:
        obj: A dataclass instance.

    Returns:
        A dictionary mapping from `obj` field names to values.
    """
    d = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return d


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


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """A trajectory, e.g. a one episode rollout from an expert policy."""

    obs: np.ndarray
    """Observations, shape (trajectory_len + 1, ) + observation_shape."""

    acts: np.ndarray
    """Actions, shape (trajectory_len, ) + action_shape."""

    infos: Optional[np.ndarray]
    """An array of info dicts, length trajectory_len."""

    terminal: bool
    """Does this trajectory (fragment) end in a terminal state?

    Episodes are always terminal. Trajectory fragments are also terminal when they
    contain the final state of an episode (even if missing the start of the episode).
    """

    def __len__(self) -> int:
        """Returns number of transitions, equal to the number of actions."""
        return len(self.acts)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Trajectory):
            return False

        dict_self, dict_other = dataclasses.asdict(self), dataclasses.asdict(other)
        # Trajectory objects may still have different keys if different subclasses
        if dict_self.keys() != dict_other.keys():
            return False

        if len(self) != len(other):
            # Short-circuit: if trajectories are of different length, then unequal.
            # Redundant as later checks would catch this, but speeds up common case.
            return False

        for k, self_v in dict_self.items():
            other_v = dict_other[k]
            if k == "infos":
                # Treat None equivalent to sequence of empty dicts
                self_v = [{}] * len(self) if self_v is None else self_v
                other_v = [{}] * len(other) if other_v is None else other_v
            if not np.array_equal(self_v, other_v):
                return False

        return True

    def __post_init__(self):
        """Performs input validation: check shapes are as specified in docstring."""
        if len(self.obs) != len(self.acts) + 1:
            raise ValueError(
                "expected one more observations than actions: "
                f"{len(self.obs)} != {len(self.acts)} + 1",
            )
        if self.infos is not None and len(self.infos) != len(self.acts):
            raise ValueError(
                "infos when present must be present for each action: "
                f"{len(self.infos)} != {len(self.acts)}",
            )
        if len(self.acts) == 0:
            raise ValueError("Degenerate trajectory: must have at least one action.")

    def __setstate__(self, state):
        if "terminal" not in state:
            warnings.warn(
                "Loading old version of Trajectory."
                "Support for this will be removed in future versions.",
                DeprecationWarning,
            )
            state["terminal"] = True
        self.__dict__.update(state)


def _rews_validation(rews: np.ndarray, acts: np.ndarray):
    if rews.shape != (len(acts),):
        raise ValueError(
            "rewards must be 1D array, one entry for each action: "
            f"{rews.shape} != ({len(acts)},)",
        )
    if not np.issubdtype(rews.dtype, np.floating):
        raise ValueError(f"rewards dtype {rews.dtype} not a float")


@dataclasses.dataclass(frozen=True, eq=False)
class TrajectoryWithRew(Trajectory):
    """A `Trajectory` that additionally includes reward information."""

    rews: np.ndarray
    """Reward, shape (trajectory_len, ). dtype float."""

    def __post_init__(self):
        """Performs input validation, including for rews."""
        super().__post_init__()
        _rews_validation(self.rews, self.acts)


Pair = Tuple[T, T]
TrajectoryPair = Pair[Trajectory]
TrajectoryWithRewPair = Pair[TrajectoryWithRew]


def transitions_collate_fn(
    batch: Sequence[Mapping[str, np.ndarray]],
) -> Mapping[str, Union[np.ndarray, th.Tensor]]:
    """Custom `torch.utils.data.DataLoader` collate_fn for `TransitionsMinimal`.

    Use this as the `collate_fn` argument to `DataLoader` if using an instance of
    `TransitionsMinimal` as the `dataset` argument.

    Args:
        batch: The batch to collate.

    Returns:
        A collated batch. Uses Torch's default collate function for everything
        except the "infos" key. For "infos", we join all the info dicts into a
        list of dicts. (The default behavior would recursively collate every
        info dict into a single dict, which is incorrect.)
    """
    batch_no_infos = [
        {k: np.array(v) for k, v in sample.items() if k != "infos"} for sample in batch
    ]

    result = th_data.dataloader.default_collate(batch_no_infos)
    assert isinstance(result, dict)
    result["infos"] = [sample["infos"] for sample in batch]
    return result


TransitionsMinimalSelf = TypeVar("TransitionsMinimalSelf", bound="TransitionsMinimal")


@dataclasses.dataclass(frozen=True)
class TransitionsMinimal(th_data.Dataset, Sequence[Mapping[str, np.ndarray]]):
    """A Torch-compatible `Dataset` of obs-act transitions.

    This class and its subclasses are usually instantiated via
    `imitation.data.rollout.flatten_trajectories`.

    Indexing an instance `trans` of TransitionsMinimal with an integer `i`
    returns the `i`th `Dict[str, np.ndarray]` sample, whose keys are the field
    names of each dataclass field and whose values are the ith elements of each field
    value.

    Slicing returns a possibly empty instance of `TransitionsMinimal` where each
    field has been sliced.
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

    def __len__(self) -> int:
        """Returns number of transitions. Always positive."""
        return len(self.obs)

    def __post_init__(self):
        """Performs input validation: check shapes & dtypes match docstring.

        Also make array values read-only.

        Raises:
            ValueError: if batch size (array length) is inconsistent
                between `obs`, `acts` and `infos`.
        """
        for val in vars(self).values():
            if isinstance(val, np.ndarray):
                val.setflags(write=False)

        if len(self.obs) != len(self.acts):
            raise ValueError(
                "obs and acts must have same number of timesteps: "
                f"{len(self.obs)} != {len(self.acts)}",
            )

        if len(self.infos) != len(self.obs):
            raise ValueError(
                "obs and infos must have same number of timesteps: "
                f"{len(self.obs)} != {len(self.infos)}",
            )

    # TODO(adam): uncomment below once pytype bug fixed in
    # issue https://github.com/google/pytype/issues/1108
    # @overload
    # def __getitem__(self: T, key: slice) -> T:
    #     pass  # pragma: no cover
    #
    # @overload
    # def __getitem__(self, key: int) -> Mapping[str, np.ndarray]:
    #     pass  # pragma: no cover

    @overload
    def __getitem__(self, key: int) -> Mapping[str, np.ndarray]:
        pass

    @overload
    def __getitem__(self: TransitionsMinimalSelf, key: slice) -> TransitionsMinimalSelf:
        pass

    def __getitem__(self, key):
        """See TransitionsMinimal docstring for indexing and slicing semantics."""
        d = dataclass_quick_asdict(self)
        d_item = {k: v[key] for k, v in d.items()}

        if isinstance(key, slice):
            # Return type is the same as this dataclass. Replace field value with
            # slices.
            return dataclasses.replace(self, **d_item)
        else:
            assert isinstance(key, int)
            # Return type is a dictionary. Array values have no batch dimension.
            #
            # Dictionary of np.ndarray values is a convenient
            # torch.util.data.Dataset return type, as a torch.util.data.DataLoader
            # taking in this `Dataset` as its first argument knows how to
            # automatically concatenate several dictionaries together to make
            # a single dictionary batch with `torch.Tensor` values.
            return d_item


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
                f"{self.obs.shape} != {self.next_obs.shape}",
            )
        if self.obs.dtype != self.next_obs.dtype:
            raise ValueError(
                "obs and next_obs must have the same dtype: "
                f"{self.obs.dtype} != {self.next_obs.dtype}",
            )
        if self.dones.shape != (len(self.acts),):
            raise ValueError(
                "dones must be 1D array, one entry for each timestep: "
                f"{self.dones.shape} != ({len(self.acts)},)",
            )
        if self.dones.dtype != bool:
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

        class TrajectoryDatasetSequence(Sequence[Trajectory]):
            """A wrapper to present a HF dataset as a sequence of trajectories."""

            def __init__(self, dataset: datasets.Dataset):
                self._dataset = dataset
                self._trajectory_class = (
                    TrajectoryWithRew if "rews" in dataset.features else Trajectory
                )

            def __len__(self) -> int:
                return len(self._dataset)

            def __getitem__(self, idx):
                if isinstance(idx, slice):
                    dataslice = self._dataset[idx]
                    trajectory_kwargs = [
                        {key: dataslice[key][i] for key in dataslice}
                        for i in range(len(dataslice["obs"]))
                    ]
                    return [
                        self._trajectory_class(**kwargs) for kwargs in trajectory_kwargs
                    ]
                else:
                    return self._trajectory_class(**self._dataset[idx])

        return TrajectoryDatasetSequence(dataset)

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


def save(path: AnyPath, trajectories: Sequence[Trajectory]):
    """Save a sequence of Trajectories to disk using HuggingFace's datasets library.

    The dataset has the following fields:
    * obs: The observations. Shape: (num_timesteps, obs_dim). dtype: float.
    * acts: The actions. Shape: (num_timesteps, act_dim). dtype: float.
    * infos: The infos. Shape: (num_timesteps, ). dtype: dict.
    * terminal: The terminal flags. Shape: (num_timesteps, ). dtype: bool.
    * rews: The rewards. Shape: (num_timesteps, ). dtype: float. if applicable.

    Args:
        path: Trajectories are saved to this path.
        trajectories: The trajectories to save.

    Raises:
        ValueError: If not all trajectories have the same type, i.e. some are
            `Trajectory` and others are `TrajectoryWithRew`.
    """
    p = parse_path(path)

    trajectory_dict = {
        "obs": [traj.obs for traj in trajectories],
        "acts": [traj.acts for traj in trajectories],
        # Replace 'None' values for `infos`` with array of empty dicts
        "infos": [
            traj.infos if traj.infos is not None else np.full(len(traj), {})
            for traj in trajectories
        ],
        "terminal": [traj.terminal for traj in trajectories],
    }
    has_reward = [isinstance(traj, TrajectoryWithRew) for traj in trajectories]
    if all(has_reward):
        trajectory_dict["rews"] = [
            cast(TrajectoryWithRew, traj).rews for traj in trajectories
        ]
    elif any(has_reward):
        raise ValueError("Some trajectories have rewards but not all")

    datasets.Dataset.from_dict(trajectory_dict).save_to_disk(p)
    logging.info(f"Dumped demonstrations to {p}.")
