"""Types and helper methods for transitions and trajectories."""

import collections
import dataclasses
import os
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import torch as th
from torch.utils import data as th_data

T = TypeVar("T")

AnyPath = Union[str, bytes, os.PathLike]


@dataclasses.dataclass(frozen=True)
class DictObs:
    # TODO: Docs!
    """
    Stores observations from an environment with a dictionary observation space.
    This class serves two purposes:
        1. enforcing invariants on the observations
        2. providing an interface that more closely reflects that of observations
        stored in a numpy array.

    Observations are in the format dict[str, np.ndarray].
    DictObs enforces that:
        - all arrays have equal first dimension. This dimension usually represents
        timesteps, but sometimes represents different environments in a vecenv.

    For the inteface, DictObs provides:
        - len(DictObs) returns the first dimension of the arrays(enforced to be equal)
        - slicing/indexing along this first dimension (returning a dictobs)
        - iterating (yeilds a series of dictobs, iterating over first dimension of each array)

    There are some other convenience functions for mapping / stacking / concatenating
    lists of dictobs.
    """

    # TODO: should support th.tensor?
    d: dict[str, np.ndarray]

    def __len__(self):
        lens = set(len(v) for v in self.d.values())
        if len(lens) == 1:
            return lens.pop()
        else:
            raise ValueError(f"observations of conflicting lengths found: {lens}")

    @classmethod
    def map(cls, obj: Union["DictObs", np.ndarray, th.Tensor], fn: Callable):
        if isinstance(obj, cls):
            return cls({k: fn(v) for k, v in obj.d.items()})
        else:
            return fn(obj)

    @classmethod
    def stack(cls, dictobs_list: Iterable["DictObs"]) -> "DictObs":
        unstacked: dict[str, list[np.ndarray]] = collections.defaultdict(list)
        for do in dictobs_list:
            for k, array in do.d.items():
                unstacked[k].append(array)
        stacked = {k: np.stack(arr_list) for k, arr_list in unstacked.items()}
        return cls(stacked)

    @classmethod
    def concatenate(cls, dictobs_list: Iterable["DictObs"]) -> "DictObs":
        unstacked: dict[str, list[np.ndarray]] = collections.defaultdict(list)
        for do in dictobs_list:
            for k, array in do.d.items():
                unstacked[k].append(array)
        catted = {
            k: np.concatenate(arr_list, axis=0) for k, arr_list in unstacked.items()
        }
        return cls(catted)

    def __getitem__(self, key: Union[slice, int]) -> "DictObs":
        # TODO assert just one slice? (multi dimensional slices are sketchy)
        # TODO test
        # TODO compare to adam's below

        # TODO -- what if you slice for a single timestep? This would presumably invalidate
        # the equal-first-dimension invariant. Do we want that invariant?
        # maybe it should be optional at least?
        return DictObs.map(self, lambda a: a[key])

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def shape(self) -> dict[str, tuple[int, ...]]:
        return {k: v.shape for k, v in self.d.items()}

    def dtype(self) -> dict[str, tuple[int, ...]]:
        return {k: v.dtype for k, v in self.d.items()}

    def unwrap(self) -> dict:
        return self.d

    @classmethod
    def maybe_wrap(
        cls, obs: Union[dict[str, np.ndarray], np.ndarray, "DictObs"]
    ) -> Union["DictObs", np.ndarray]:
        """If `obs` is a dict, wraps in a dict obs.
        If `obs` is an array or already an obsdict, returns it unchanged"""
        if isinstance(obs, dict):
            return cls(obs)
        else:
            assert isinstance(obs, (np.ndarray, cls))
            return obs

    @classmethod
    def maybe_unwrap(
        cls, maybe_dictobs: Union["DictObs", np.ndarray]
    ) -> Union[dict[str, np.ndarray], np.ndarray]:
        if isinstance(maybe_dictobs, cls):
            return maybe_dictobs.unwrap()
        else:
            assert isinstance(maybe_dictobs, (np.ndarray, th.Tensor))
            return maybe_dictobs

    # TODO: eq
    # TODO: post_init len check?


def assert_not_dictobs(x: Union[np.ndarray, DictObs]) -> np.ndarray:
    if isinstance(x, DictObs):
        raise ValueError("Dictionary observations are not supported here.")
    return x


# TODO: maybe should support DictObs?
TransitionMapping = Mapping[str, Union[np.ndarray, th.Tensor]]


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


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """A trajectory, e.g. a one episode rollout from an expert policy."""

    obs: Union[np.ndarray, DictObs]
    """Observations, shape (trajectory_len + 1, ) + observation_shape."""

    acts: np.ndarray
    """Actions, shape (trajectory_len, ) + action_shape."""

    infos: Optional[np.ndarray]
    """An array of info dicts, shape (trajectory_len, ).

    The info dict is returned by some environments `step()` and contains auxiliary
    diagnostic information. For example the monitor wrapper adds an info dict
    to the last step of each episode containing the episode return and length.
    """

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
        {k: np.array(v) for k, v in sample.items() if k in ["acts", "dones"]}
        for sample in batch
    ]

    result = th_data.dataloader.default_collate(batch_no_infos)
    assert isinstance(result, dict)
    result["infos"] = [sample["infos"] for sample in batch]
    result["obs"] = stack([sample["obs"] for sample in batch])
    result["next_obs"] = stack([sample["next_obs"] for sample in batch])
    # TODO: clean names, docs
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

    obs: Union[np.ndarray, DictObs]
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

    next_obs: Union[np.ndarray, DictObs]
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
        # TODO: could add support for checking dictobs shape/dtype of each array
        # would be nice for debugging to have a dictobs.shapes -> dict{str: shape}
        if isinstance(self.obs, np.ndarray):
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


ObsType = TypeVar("ObsType", np.ndarray, DictObs)


def concatenate(arrs: List[ObsType]) -> ObsType:
    assert len(arrs) > 0
    if isinstance(arrs[0], DictObs):
        return DictObs.concatenate(arrs)
    else:
        return np.concatenate(arrs)


def stack(arrs: List[ObsType]) -> ObsType:
    assert len(arrs) > 0
    if isinstance(arrs[0], DictObs):
        return DictObs.stack(arrs)
    else:
        return np.stack(arrs)


# class ObsList(collections.UserList[O]):
#     def isDict(self) -> bool:
#         return (len(self) > 0) and isinstance(self[0], DictObs)

#     def concatenate(self) -> O:
#         if self.isDict():
#             return DictObs.concatenate(self)
#         else:
#             return np.concatenate(self)

#     def stack(self) -> O:
#         if self.isDict():
#             return DictObs.concatenate(self)
#         else:
#             return np.concatenate(self)
