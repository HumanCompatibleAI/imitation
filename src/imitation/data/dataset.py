import abc
import dataclasses
from typing import Dict, Generic, Mapping, Optional, Tuple, Type, TypeVar

import numpy as np

from imitation.data import types

T = TypeVar("T")


class Dataset(abc.ABC, Generic[T]):
    def sample(self, n_samples: int) -> T:
        """Return a batch of data.

        Args:
            n_samples: A positive integer indicating the number of samples to return.
        Raises:
            ValueError: If n_samples is nonpositive.
        """

    def size(self) -> Optional[int]:
        """Number of samples in this dataset, ie the epoch size."""
        return None


class DictDataset(Dataset[Dict[str, np.ndarray]]):
    def __init__(self, data_map: Mapping[str, np.ndarray]):
        """Abstract base class for sampling data from an in-memory dictionary.

        The return value of `.sample(n_samples)` is a dictionary with the same keys as
        `data_map` and whose values are `n_samples` stacked rows selected from the
        values of `data_map`. In the return value, rows that were previously parallel
        remain parallel. As an example, if
        `data_map=dict(a=np.arange(5), b=(np.arange(5) + 4))`, then it is guaranteed
        that every call to `.sample()` returns a dictionary `batch` such that
        `batch['a'] == batch['b'] + 4`.

        Args:
            data_map: A mapping from keys to np.ndarray values, where every value has
                an equal number of rows. This argument is not never mutated, as each
                array is immediately copied. Required to be non-empty.
        """
        if len(data_map) == 0:
            raise ValueError("Empty data_map not allowed.")
        self.data_map = {k: v.copy() for k, v in data_map.items()}
        n_samples_set = set(len(v) for v in data_map.values())

        if len(n_samples_set) != 1:
            raise ValueError(
                f"Unequal number of rows in data_map values: {n_samples_set}"
            )
        self._n_data = next(iter(n_samples_set))

    def size(self):
        return self._n_data


class EpochOrderDictDataset(DictDataset):
    def __init__(self, data_map: Mapping[str, np.ndarray], shuffle: bool = True):
        """In-memory data sampler that samples in epoch-order.

        No sample from `data_map` can be returned an X+1th time by `sample()` until
        every other sample has been returned X times.

        Args:
            data_map: A mapping from keys to np.ndarray values, where every value has
                an equal number of rows. This argument is not mutated, as each array
                is immediately copied.
            shuffle: If true, then shuffle the dataset upon initialization and at the
                end of every epoch.
        """
        super().__init__(data_map)
        self._next_idx = 0
        self._shuffle = shuffle
        if shuffle:
            self.shuffle_dataset()

    def shuffle_dataset(self):
        """Shuffles the data_map in place."""
        perm = np.arange(self.size())
        np.random.shuffle(perm)
        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

    def sample(self, n_samples: int) -> Dict[str, np.ndarray]:
        if n_samples <= 0:
            raise ValueError(f"n_samples={n_samples} must be positive.")
        samples_accum = {k: [] for k in self.data_map.keys()}
        n_samples_remain = n_samples
        while n_samples_remain > 0:
            n_samples_actual, sample = self._sample_bounded(n_samples_remain)
            n_samples_remain -= n_samples_actual
            for k, v in sample.items():
                samples_accum[k].append(v)

        result = {k: np.concatenate(v) for k, v in samples_accum.items()}
        assert all(len(v) == n_samples for v in result.values())
        return result

    def _sample_bounded(
        self, n_samples_request: int
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        """Like `.sample()`, but allowed to return fewer samples on epoch boundaries."""
        assert n_samples_request > 0
        if self._next_idx >= self.size():
            self._next_idx = 0
            if self._shuffle:
                self.shuffle_dataset()

        n_samples_actual = min(n_samples_request, self.size() - self._next_idx)
        assert n_samples_actual > 0

        result = {}
        for key in self.data_map:
            result[key] = self.data_map[key][
                self._next_idx : self._next_idx + n_samples_actual
            ]
            assert len(result[key]) == n_samples_actual
        self._next_idx += n_samples_actual
        return n_samples_actual, result


class RandomDictDataset(EpochOrderDictDataset):
    """In-memory data sampler that uniformly samples with replacement."""

    def sample(self, n_samples: int) -> Dict[str, np.ndarray]:
        if n_samples <= 0:
            raise ValueError(f"n_samples={n_samples} must be positive.")
        inds = np.random.randint(self.size(), size=n_samples)
        return {k: v[inds] for k, v in self.data_map.items()}


S = TypeVar("S", bound=type(types.Transitions))  # Must be subclass of Transitions


class TransitionsDictDatasetAdaptor(Dataset[S]):
    def __init__(
        self,
        transitions: S,
        simple_dataset_cls: Type[DictDataset] = RandomDictDataset,
        simple_dataset_cls_kwargs: Optional[Mapping] = None,
    ):
        data_map: Dict[str, np.ndarray] = dataclasses.asdict(transitions)
        kwargs = simple_dataset_cls_kwargs or {}
        self.transitions_cls: Type[S] = type(transitions)
        self.simple_dataset = simple_dataset_cls(data_map, **kwargs)

    def sample(self, n_samples) -> S:
        dict_samples = self.simple_dataset.sample(n_samples)
        result = self.transitions_cls(**dict_samples)
        assert len(result) == n_samples
        return result

    def size(self):
        return self.simple_dataset.size()
