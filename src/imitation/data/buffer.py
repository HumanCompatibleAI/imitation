import dataclasses
from typing import Dict, Optional, Tuple

import numpy as np
from stable_baselines.common import vec_env

from imitation.data import types


class Buffer:
    """A FIFO ring buffer for NumPy arrays of a fixed shape and dtype.

    Supports random sampling with replacement.
    """

    capacity: int
    """The number of data samples that can be stored in this buffer."""

    sample_shapes: Dict[str, Tuple[int, ...]]
    """The shapes of each data sample stored in this buffer."""

    _arrays: Dict[str, np.ndarray]
    """The underlying NumPy arrays (which actually store the data)."""

    _n_data: int
    """The number of samples currently stored in this buffer.

  An integer in `range(0, self.capacity + 1)`. This attribute is the return
  value of `self.__len__`.
  """

    _idx: int
    """The index of the first row that new data should be written to.

  An integer in `range(0, self.capacity)`.
  """

    def __init__(
        self,
        capacity: int,
        sample_shapes: Dict[str, Tuple[int, ...]],
        dtypes: Dict[str, np.dtype],
    ):
        """Constructs a Buffer.

        Args:
            capacity: The number of samples that can be stored.
            sample_shapes: A dictionary mapping string keys to the shape of
                samples associated with that key.
            dtypes (`np.dtype`-like): A dictionary mapping string keys to the dtype
                of samples associated with that key.

        Raises:
            KeyError: `sample_shapes` and `dtypes` have different keys.
        """
        if sample_shapes.keys() != dtypes.keys():
            raise KeyError("sample_shape and dtypes keys don't match")
        self.capacity = capacity
        self.sample_shapes = {k: tuple(shape) for k, shape in sample_shapes.items()}
        self._arrays = {
            k: np.zeros((capacity,) + shape, dtype=dtypes[k])
            for k, shape in self.sample_shapes.items()
        }
        self._n_data = 0
        self._idx = 0

    @classmethod
    def from_data(cls, data: Dict[str, np.ndarray]) -> "Buffer":
        """Constructs and return a Buffer containing only the provided data.

        The returned Buffer is at full capacity and ready for sampling.

        Args:
            data: A dictionary mapping keys to data arrays. The arrays may differ
                in their shape, but should agree in the first axis.

        Raises:
            ValueError: `data` is empty.
            ValueError: `data` has items mapping to arrays differing in the
                length of their first axis.
        """
        capacities = [arr.shape[0] for arr in data.values()]
        capacities = np.unique(capacities)
        if len(data) == 0:
            raise ValueError("No keys in data.")
        if len(capacities) > 1:
            raise ValueError("Keys map to different length values")
        capacity = capacities[0]

        sample_shapes = {k: arr.shape[1:] for k, arr in data.items()}
        dtypes = {k: arr.dtype for k, arr in data.items()}
        buf = cls(capacity, sample_shapes, dtypes)
        buf.store(data)
        return buf

    def store(self, data: Dict[str, np.ndarray]) -> None:
        """Stores new data samples, replacing old samples with FIFO priority.

        Args:
            data: A dictionary mapping keys `k` to arrays with shape
                `(n_samples,) + self.sample_shapes[k]`, where `n_samples` is less
                than or equal to `self.capacity`.

        Raises:
            ValueError: `data` is empty.
            ValueError: If `n_samples` is greater than `self.capacity`.
            ValueError: data is the wrong shape.
        """
        expected_keys = set(self.sample_shapes.keys())
        missing_keys = expected_keys.difference(data.keys())
        unexpected_keys = set(data.keys()).difference(expected_keys)
        if len(missing_keys) > 0:
            raise ValueError(f"Missing keys {missing_keys}")
        if len(unexpected_keys) > 0:
            raise ValueError(f"Unexpected keys {unexpected_keys}")

        n_samples = [arr.shape[0] for arr in data.values()]
        n_samples = np.unique(n_samples)
        if len(n_samples) > 1:
            raise ValueError("Keys map to different length values.")
        n_samples = n_samples[0]

        if n_samples == 0:
            raise ValueError("Trying to store empty data.")
        if n_samples > self.capacity:
            raise ValueError("Not enough capacity to store data.")

        for k, arr in data.items():
            if arr.shape[1:] != self.sample_shapes[k]:
                raise ValueError(f"Wrong data shape for {k}")

        new_idx = self._idx + n_samples
        if new_idx > self.capacity:
            n_remain = self.capacity - self._idx
            # Need to loop around the buffer. Break into two "easy" calls.
            self._store_easy({k: arr[:n_remain] for k, arr in data.items()})
            assert self._idx == 0
            self._store_easy({k: arr[n_remain:] for k, arr in data.items()})
        else:
            self._store_easy(data)

    def _store_easy(self, data: Dict[str, np.ndarray]) -> None:
        """Stores new data samples, replacing old samples with FIFO priority.

        Requires that `len(data) <= self.capacity - self._idx`. Updates `self._idx`
        to be the insertion point of the next call to `_store_easy` call,
        looping back to `self._idx = 0` if necessary.

        Also updates `self._n_data`.

        Args:
            data: Same as in `self.store`'s docstring, except with the additional
                constraint `len(data) <= self.capacity - self._idx`.
        """
        n_samples = [arr.shape[0] for arr in data.values()]
        n_samples = np.unique(n_samples)
        assert len(n_samples) == 1
        n_samples = n_samples[0]

        assert n_samples <= self.capacity - self._idx
        idx_hi = self._idx + n_samples
        for k, arr in data.items():
            self._arrays[k][self._idx : idx_hi] = arr
        self._idx = idx_hi % self.capacity
        self._n_data = min(self._n_data + n_samples, self.capacity)

    def sample(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Uniformly sample `n_samples` samples from the buffer with replacement.

        Args:
            n_samples: The number of samples to randomly sample.

        Returns:
            samples (np.ndarray): An array with shape
                `(n_samples) + self.sample_shape`.

        Raises:
            ValueError: The buffer is empty.
        """
        if len(self) == 0:
            raise ValueError("Buffer is empty")
        ind = np.random.randint(len(self), size=n_samples)
        return {k: buffer[ind] for k, buffer in self._arrays.items()}

    def __len__(self) -> int:
        """Returns the number of samples stored in the buffer."""
        assert 0 <= self._n_data <= self.capacity
        return self._n_data


class ReplayBuffer:
    """Buffer for Transitions."""

    capacity: int
    """The number of data samples that can be stored in this buffer."""

    def __init__(
        self,
        capacity: int,
        venv: Optional[vec_env.VecEnv] = None,
        *,
        obs_shape: Optional[Tuple[int, ...]] = None,
        act_shape: Optional[Tuple[int, ...]] = None,
        obs_dtype: Optional[np.dtype] = None,
        act_dtype: Optional[np.dtype] = None,
    ):
        """Constructs a ReplayBuffer.

        Args:
            capacity: The number of samples that can be stored.
            venv: The environment whose action and observation
                spaces can be used to determine the data shapes of the underlying
                buffers. Overrides all the following arguments.
            obs_shape: The shape of the observation space.
            act_shape: The shape of the action space.
            obs_dtype: The dtype of the observation space.
            act_dtype: The dtype of the action space.

        Raises:
            ValueError: Couldn't infer the observation and action shapes and dtypes
                from the arguments.
        """
        params = [obs_shape, act_shape, obs_dtype, act_dtype]
        if venv is not None:
            if np.any([x is not None for x in params]):
                raise ValueError("Specified shape or dtype and environment.")
            obs_shape = tuple(venv.observation_space.shape)
            act_shape = tuple(venv.action_space.shape)
            obs_dtype = venv.observation_space.dtype
            act_dtype = venv.action_space.dtype
        else:
            if np.any([x is None for x in params]):
                raise ValueError("Shape or dtype missing and no environment specified.")

        self.capacity = capacity
        sample_shapes = {
            "obs": obs_shape,
            "acts": act_shape,
            "next_obs": obs_shape,
            "dones": (),
        }
        dtypes = {
            "obs": obs_dtype,
            "acts": act_dtype,
            "next_obs": obs_dtype,
            "dones": np.bool,
        }
        self._buffer = Buffer(capacity, sample_shapes=sample_shapes, dtypes=dtypes)

    @classmethod
    def from_data(cls, transitions: types.Transitions) -> "ReplayBuffer":
        """Construct and return a ReplayBuffer containing only the provided data.

        The returned ReplayBuffer is at full capacity and ready for sampling.

        Args:
            transitions: Transitions to store.

        Returns:
            A new ReplayBuffer.
        """
        capacity, *obs_shape = transitions.obs.shape
        _, *act_shape = transitions.acts.shape
        instance = cls(
            capacity=capacity,
            obs_shape=obs_shape,
            act_shape=act_shape,
            obs_dtype=transitions.obs.dtype,
            act_dtype=transitions.acts.dtype,
        )
        instance.store(transitions)
        return instance

    def sample(self, n_samples: int) -> types.Transitions:
        """Sample obs-act-obs triples.

        Args:
            n_samples: The number of samples.

        Returns:
            A Transitions named tuple containing n_samples transitions.
        """
        sample = self._buffer.sample(n_samples)
        return types.Transitions(**sample)

    def store(self, transitions: types.Transitions, truncate_ok: bool = True,) -> None:
        """Store obs-act-obs triples.

        Args:
          transitions: Transitions to store.
          truncate_ok: If False, then error if the length of `transitions` is
            greater than `self.capacity`. Otherwise, store only the final
            `self.capacity` transitions.

        Raises:
            ValueError: The arguments didn't have the same length.
        """
        trans_dict = dataclasses.asdict(transitions)
        # Remove unnecessary fields
        trans_dict = {k: trans_dict[k] for k in self._buffer.sample_shapes.keys()}

        if len(transitions) > self.capacity and truncate_ok:
            trans_dict = {k: v[-self.capacity :] for k, v in trans_dict.items()}

        self._buffer.store(trans_dict)

    def __len__(self):
        return len(self._buffer)
