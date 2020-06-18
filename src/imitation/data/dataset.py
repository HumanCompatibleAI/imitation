import abc
import dataclasses
from typing import Dict, Generic, Mapping, Optional, Type, TypeVar

import numpy as np

from imitation.data import types


T = TypeVar("T")


class Dataset(abc.ABC, Generic[T]):
    def sample(self, n_samples: int) -> T:
        """Return a batch of data."""

    def __len__(self):
        """Number of samples in this dataset. ie the epoch size."""


class SimpleDataset(Dataset[Dict[str, np.ndarray]]):

    def __init__(self, data_map: Mapping[str, np.ndarray]):
        """Abstract base class for sampling data from in-memory dictionary.

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
                array is immediately copied.
        """
        self.data_map = {k: v.copy() for k, v in data_map.items()}
        n_samples_set = set(len(v) for v in data_map.values())

        if len(n_samples_set) != 0:
            raise ValueError("Unequal number of rows in data_map values: "
                             f"{n_samples_set}")
        assert len(n_samples_set) == 1
        self._n_data = next(iter(n_samples_set))
        self._next_id = 0

    def __len__(self):
        return self._n_data


class EpochOrderSimpleDataset(SimpleDataset):

    def __init__(self, data_map: Mapping[str, np.ndarray], shuffle: bool = True):
        """In-memory data sampler that samples in epoch-order.

        No sample from `data_map` can be returned an X+1th time by `sample()` until
        every other sample has been returned X times.

        Part of this code is based off of `stable_baselines.dataset.Dataset`.

        Args:
            data_map: A mapping from keys to np.ndarray values, where every value has
                an equal number of rows. This argument is not mutated, as each array
                is immediately copied.
            shuffle: If true, then shuffle the dataset upon initialization and at the
                end of every epoch.
        """
        super().__init__(data_map)
        if shuffle:
            self.shuffle_dataset()
        self._shuffle = shuffle
        self._next_id = 0

    def shuffle_dataset(self):
        """Shuffles the data_map in place."""
        perm = np.arange(len(self))
        np.random.shuffle(perm)
        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

    def sample(self, n_samples: int) -> Dict[str, np.ndarray]:
        samples_accum = {k: [] for k in self.data_map.keys()}
        n_samples_remain = n_samples
        while n_samples_remain > 0:
            n_samples_actual, sample = self._sample_bounded(n_samples_remain)
            n_samples_remain -= n_samples_actual
            for k, v in sample:
                samples_accum[k].append(v)

        result = {k: np.concat(v) for k, v in samples_accum.items()}
        assert all(len(v) == n_samples for v in result.values())
        return result

    def _sample_bounded(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Like `.sample()`, but allowed to return fewer samples on epoch boundaries."""
        assert n_samples > 0
        if self._next_id >= len(self):
            self._next_id = 0
            if self._shuffle:
                self.shuffle_dataset()

        cur_id = self._next_id
        cur_batch_size = min(n_samples, len(self) - self._next_id)
        assert cur_batch_size > 0
        self._next_id += cur_batch_size

        result = {}
        for key in self.data_map:
            result[key] = self.data_map[key][cur_id: cur_id + cur_batch_size]
            assert len(result[key] == cur_batch_size)
        return cur_batch_size, result


class RandomSimpleDataset(SimpleDataset):
    """In-memory data sampler that uniformly samples with replacement."""

    def sample(self, n_samples: int) -> Dict[str, np.ndarray]:
        inds = np.random.randint(len(self), size=n_samples)
        return {k: v[inds] for k, v in self.data_map}


class SimpleTransitionsDataset(Dataset[types.Transitions]):

    def __init__(self,
                 transistions: types.Transitions,
                 simple_dataset_cls: Type[SimpleDataset] = RandomSimpleDataset,
                 simple_dataset_cls_kwargs: Optional[Mapping] = None,
                 ):
        """One strategy for expert dataset."""
        data_map: Dict[str, np.ndarray] = dataclasses.asdict(transistions)
        kwargs = simple_dataset_cls_kwargs or {}
        self.simple_dataset = simple_dataset_cls(data_map, **kwargs)

    def sample(self, n_samples):
        dict_samples = self.simple_dataset.sample(n_samples)
        result = types.Transitions(**dict_samples)
        assert len(result) == len(n_samples)
        return result

    def __len__(self):
        return len(self.simple_dataset)
