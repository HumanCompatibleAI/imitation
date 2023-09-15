"""Buffers to store NumPy arrays and transitions in."""

from typing import Any, Mapping, Optional, Tuple

import numpy as np
from stable_baselines3.common import vec_env

from imitation.data import types


def num_samples(data: Mapping[Any, np.ndarray]) -> int:
    """Computes the number of samples contained in `data`.

    Args:
        data: A Mapping from keys to NumPy arrays.

    Returns:
        The unique length of the first dimension of arrays contained in `data`.

    Raises:
        ValueError: The length is not unique.
    """
    n_samples_list = [arr.shape[0] for arr in data.values()]
    n_samples_np = np.unique(n_samples_list)
    if len(n_samples_np) > 1:
        raise ValueError("Keys map to different length values.")
    return int(n_samples_np[0])


class Buffer:
    """A FIFO ring buffer for NumPy arrays of a fixed shape and dtype.

    Supports random sampling with replacement.
    """

    capacity: int
    """The number of data samples that can be stored in this buffer."""

    sample_shapes: Mapping[str, Tuple[int, ...]]
    """The shapes of each data sample stored in this buffer."""

    _arrays: Mapping[str, np.ndarray]
    """The underlying NumPy arrays (which actually store the data)."""

    _n_data: int
    """The number of samples currently stored in this buffer.

    An integer in `range(0, self.capacity + 1)`. This attribute is the return
    value of `self.size()`.
    """

    _idx: int
    """The index of the first row that new data should be written to.

    An integer in `range(0, self.capacity)`.
    """

    def __init__(
        self,
        capacity: int,
        sample_shapes: Mapping[str, Tuple[int, ...]],
        dtypes: Mapping[str, np.dtype],
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
    def from_data(
        cls,
        data: Mapping[str, np.ndarray],
        capacity: Optional[int] = None,
        truncate_ok: bool = False,
    ) -> "Buffer":
        """Constructs and return a Buffer containing the provided data.

        Shapes and dtypes are automatically inferred.

        Args:
            data: A dictionary mapping keys to data arrays. The arrays may differ
                in their shape, but should agree in the first axis.
            capacity: The Buffer capacity. If not provided, then this is automatically
                set to the size of the data, so that the returned Buffer is at full
                capacity.
            truncate_ok: Whether to error if `capacity` < the number of samples in
                `data`. If False, then only store the last `capacity` samples from
                `data` when overcapacity.

        Examples:
            In the follow examples, suppose the arrays in `data` are length-1000.

            `Buffer` with same capacity as arrays in `data`::

                Buffer.from_data(data)

            `Buffer` with larger capacity than arrays in `data`::

                Buffer.from_data(data, 10000)

            `Buffer with smaller capacity than arrays in `data`. Without
            `truncate_ok=True`, `from_data` will error::

                Buffer.from_data(data, 5, truncate_ok=True)

        Returns:
            Buffer of specified `capacity` containing provided `data`.

        Raises:
            ValueError: `data` is empty.
            ValueError: `data` has items mapping to arrays differing in the
                length of their first axis.
        """
        data_capacities_list = [arr.shape[0] for arr in data.values()]
        data_capacities = np.unique(data_capacities_list)
        if len(data) == 0:
            raise ValueError("No keys in data.")
        if len(data_capacities) > 1:
            raise ValueError("Keys map to different length values")
        if capacity is None:
            capacity = data_capacities[0]

        sample_shapes = {k: arr.shape[1:] for k, arr in data.items()}
        dtypes = {k: arr.dtype for k, arr in data.items()}
        buf = cls(capacity, sample_shapes, dtypes)
        buf.store(data, truncate_ok=truncate_ok)
        return buf

    def store(self, data: Mapping[str, np.ndarray], truncate_ok: bool = False) -> None:
        """Stores new data samples, replacing old samples with FIFO priority.

        Args:
            data: A dictionary mapping keys `k` to arrays with shape
                `(n_samples,) + self.sample_shapes[k]`, where `n_samples` is less
                than or equal to `self.capacity`.
            truncate_ok: If False, then error if the length of `transitions` is
                greater than `self.capacity`. Otherwise, store only the final
                `self.capacity` transitions.

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

        n_samples = num_samples(data)
        if n_samples == 0:
            raise ValueError("Trying to store empty data.")
        if n_samples > self.capacity:
            if not truncate_ok:
                raise ValueError("Not enough capacity to store data.")
            else:
                data = {k: arr[-self.capacity :] for k, arr in data.items()}

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

    def _store_easy(self, data: Mapping[str, np.ndarray]) -> None:
        """Stores new data samples, replacing old samples with FIFO priority.

        Requires that `size(data) <= self.capacity - self._idx`, where `size(data)` is
        the number of rows in every array in `data.values()`. Updates `self._idx`
        to be the insertion point of the next call to `_store_easy` call,
        looping back to `self._idx = 0` if necessary.

        Also updates `self._n_data`.

        Args:
            data: Same as in `self.store`'s docstring, except with the additional
                constraint `size(data) <= self.capacity - self._idx`.
        """
        n_samples = num_samples(data)
        assert n_samples <= self.capacity - self._idx
        idx_hi = self._idx + n_samples
        for k, arr in data.items():
            self._arrays[k][self._idx : idx_hi] = arr
        self._idx = idx_hi % self.capacity
        self._n_data = min(self._n_data + n_samples, self.capacity)

    def sample(self, n_samples: int) -> Mapping[str, np.ndarray]:
        """Uniformly sample `n_samples` samples from the buffer with replacement.

        Args:
            n_samples: The number of samples to randomly sample.

        Returns:
            samples (np.ndarray): An array with shape
                `(n_samples) + self.sample_shape`.

        Raises:
            ValueError: The buffer is empty.
        """
        if self.size() == 0:
            raise ValueError("Buffer is empty")
        ind = np.random.randint(self.size(), size=n_samples)
        return {k: buffer[ind] for k, buffer in self._arrays.items()}

    def size(self) -> int:
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
                buffers. Mutually exclusive with shape and dtype arguments.
            obs_shape: The shape of the observation space.
            act_shape: The shape of the action space.
            obs_dtype: The dtype of the observation space.
            act_dtype: The dtype of the action space.

        Raises:
            ValueError: Couldn't infer the observation and action shapes and dtypes
                from the arguments.
            ValueError: Specified both venv and shapes/dtypes.
        """
        params = (obs_shape, act_shape, obs_dtype, act_dtype)
        if venv is not None:
            if not all(x is None for x in params):
                raise ValueError(
                    "Cannot specify both shape/dtype and also environment.",
                )
            obs_shape = tuple(venv.observation_space.shape)
            act_shape = tuple(venv.action_space.shape)
            obs_dtype = venv.observation_space.dtype
            act_dtype = venv.action_space.dtype
        else:
            if any(x is None for x in params):
                raise ValueError("Shape or dtype missing and no environment specified.")

        assert obs_shape is not None
        assert act_shape is not None
        assert obs_dtype is not None
        assert act_dtype is not None

        self.capacity = capacity
        sample_shapes = {
            "obs": obs_shape,
            "acts": act_shape,
            "next_obs": obs_shape,
            "dones": (),
            "infos": (),
        }
        dtypes = {
            "obs": obs_dtype,
            "acts": act_dtype,
            "next_obs": obs_dtype,
            "dones": np.dtype(bool),
            "infos": np.dtype(object),
        }
        self._buffer = Buffer(capacity, sample_shapes=sample_shapes, dtypes=dtypes)

    @classmethod
    def from_data(
        cls,
        transitions: types.Transitions,
        capacity: Optional[int] = None,
        truncate_ok: bool = False,
    ) -> "ReplayBuffer":
        """Construct and return a ReplayBuffer containing the provided data.

        Shapes and dtypes are automatically inferred, and the returned ReplayBuffer is
        ready for sampling.

        Args:
            transitions: Transitions to store.
            capacity: The ReplayBuffer capacity. If not provided, then this is
                automatically set to the size of the data, so that the returned Buffer
                is at full capacity.
            truncate_ok: Whether to error if `capacity` < the number of samples in
                `data`. If False, then only store the last `capacity` samples from
                `data` when overcapacity.

        Examples:
            `ReplayBuffer` with same capacity as arrays in `data`::

                ReplayBuffer.from_data(data)

            `ReplayBuffer` with larger capacity than arrays in `data`::

                ReplayBuffer.from_data(data, 10000)

            `ReplayBuffer with smaller capacity than arrays in `data`. Without
            `truncate_ok=True`, `from_data` will error::

                ReplayBuffer.from_data(data, 5, truncate_ok=True)

        Returns:
            A new ReplayBuffer.
        """
        obs = types.assert_not_dictobs(transitions.obs)
        obs_shape = obs.shape[1:]
        act_shape = transitions.acts.shape[1:]
        if capacity is None:
            capacity = obs.shape[0]
        instance = cls(
            capacity=capacity,
            obs_shape=obs_shape,
            act_shape=act_shape,
            obs_dtype=obs.dtype,
            act_dtype=transitions.acts.dtype,
        )
        instance.store(transitions, truncate_ok=truncate_ok)
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

    def store(self, transitions: types.Transitions, truncate_ok: bool = True) -> None:
        """Store obs-act-obs triples.

        Args:
            transitions: Transitions to store.
            truncate_ok: If False, then error if the length of `transitions` is
                greater than `self.capacity`. Otherwise, store only the final
                `self.capacity` transitions.

        Raises:
            ValueError: The arguments didn't have the same length.
        """  # noqa: DAR402
        trans_dict = types.dataclass_quick_asdict(transitions)
        # Remove unnecessary fields
        trans_dict = {k: trans_dict[k] for k in self._buffer.sample_shapes.keys()}
        self._buffer.store(trans_dict, truncate_ok=truncate_ok)

    def size(self) -> Optional[int]:
        """Returns the number of samples stored in the buffer."""
        return self._buffer.size()
