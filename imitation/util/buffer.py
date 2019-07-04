from typing import Optional, Tuple

import gym
import numpy as np


class Buffer:
  """A FIFO ring buffer for numpy arrays of a fixed shape and dtype.

  Supports random sampling with replacement.
  """

  capacity: int
  """The number of data samples that can be stored in this buffer."""

  sample_shape: Tuple[int, ...]
  """The shape of each data sample stored in this buffer."""

  _buffer: np.ndarray
  """The underlying numpy array (which actually stores the data)."""

  _n_data: int
  """The number of samples currently stored in this buffer.

  An integer in `range(0, self.capacity + 1)`. This attribute is the return
  value of `self.__len__`.
  """

  _idx: int
  """The index of the first row that new data should be written to.

  An integer in `range(0, self.capacity)`.
  """

  def __init__(self, capacity: int, sample_shape: Tuple[int, ...], dtype):
    """Constructs a Buffer.

    Args:
        capacity: The number of samples that can be stored.
        sample_shape: The shape of each sample stored in this buffer.
        dtype (`np.dtype`-like): The numpy dtype that will be stored.
    """
    self.capacity = capacity
    self.sample_shape = tuple(sample_shape)
    self._buffer = np.zeros((capacity,) + self.sample_shape, dtype=dtype)
    self._n_data = 0
    self._idx = 0

  @classmethod
  def from_data(cls, data: np.ndarray) -> "Buffer":
    """Constructs and return a Buffer containing only the provided data.

    The returned ReplayBuffer is at full capacity and ready for sampling.
    """
    capacity, *sample_shape = data.shape
    buf = cls(capacity, sample_shape, dtype=data.dtype)
    buf.store(data)
    return buf

  def store(self, data: np.ndarray) -> None:
    """Stores new data samples, replacing old samples with FIFO priority.

    Args:
        data: An array with shape `(n_samples,) + self.sample_shape`, where
            `n_samples` is less than or equal to `self.capacity`.

    Raises:
        ValueError: `data` is empty.
        ValueError: If `n_samples` is greater than `self.capacity`.
        ValueError: data is the wrong shape.
    """
    if len(data) == 0:
      raise ValueError("Trying to store empty data.")
    if len(data) > self.capacity:
      raise ValueError("Not enough capacity to store data.")
    if data.shape[1:] != self.sample_shape:
      raise ValueError("Wrong data_shape")

    new_idx = self._idx + len(data)
    if new_idx > self.capacity:
      n_remain = self.capacity - self._idx
      # Need to loop around the buffer. Break into two "easy" calls.
      self._store_easy(data[:n_remain])
      assert self._idx == 0
      self._store_easy(data[n_remain:])
    else:
      self._store_easy(data)

  def _store_easy(self, data: np.ndarray) -> None:
    """Stores new data samples, replacing old samples with FIFO priority.

    Requires that `len(data) <= self.capacity - self._idx`. Updates `self._idx`
    to be the insertion point of the next call to `_store_easy` call,
    looping back to `self._idx = 0` if necessary.

    Also updates `self._n_data`.

    Args:
        data: Same as in `self.store`'s docstring, except with the additional
            constraint `len(data) <= self.capacity - self._idx`.
    """
    assert len(data) <= self.capacity - self._idx
    idx_hi = self._idx + len(data)
    self._buffer[self._idx:idx_hi] = data
    self._idx = idx_hi % self.capacity
    self._n_data = min(self._n_data + len(data), self.capacity)

  def sample(self, n_samples: int) -> np.ndarray:
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
    return self._buffer[ind]

  def __len__(self) -> int:
    """Returns the number of samples stored in the buffer."""
    assert 0 <= self._n_data <= self.capacity
    return self._n_data


class ReplayBuffer:
  """Wrapper around 3 `Buffer`s used to store and sample obs-act-obs triples.
  """

  capacity: int
  """The number of data samples that can be stored in this buffer."""

  def __init__(self, capacity: int,
               env: Optional[gym.Env] = None, *,
               obs_shape: Optional[Tuple[int, ...]] = None,
               act_shape: Optional[Tuple[int, ...]] = None,
               obs_dtype: Optional[np.dtype] = None,
               act_dtype: Optional[np.dtype] = None):
    """Constructs a ReplayBuffer.

    Args:
        capacity: The number of samples that can be stored.
        env: The environment whose action and observation
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
    if env is not None:
      if np.any([x is not None for x in params]):
        raise ValueError("Specified shape or dtype and environment.")
      obs_shape = tuple(env.observation_space.shape)
      act_shape = tuple(env.action_space.shape)
      obs_dtype = env.observation_space.dtype
      act_dtype = env.action_space.dtype
    else:
      if np.any([x is None for x in params]):
        raise ValueError("Shape or dtype missing and no environment specified.")

    self.capacity = capacity
    self._old_obs_buffer = Buffer(capacity, obs_shape, obs_dtype)
    self._act_buffer = Buffer(capacity, act_shape, act_dtype)
    self._new_obs_buffer = Buffer(capacity, obs_shape, obs_dtype)

  @classmethod
  def from_data(cls, old_obs: np.ndarray, act: np.ndarray, new_obs: np.ndarray
                ) -> "ReplayBuffer":
    """Construct and return a ReplayBuffer containing only the provided data.

    The returned ReplayBuffer is at full capacity and ready for sampling.

    Args:
        old_obs: Old observations.
        act: Actions.
        new_obs: New observations.

    Returns:
        A new ReplayBuffer.

    Raises:
        ValueError: The arguments didn't have the same length.
        ValueError: old_obs and new_obs have a different dtype.
    """
    if not len(old_obs) == len(act) == len(new_obs):
      raise ValueError("Arguments must have the same length.")
    if old_obs.dtype != new_obs.dtype:
      raise ValueError("old_obs and new_obs must have the same dtype.")

    capacity, *obs_shape = old_obs.shape
    _, *act_shape = act.shape
    instance = cls(capacity=capacity, obs_shape=obs_shape, act_shape=act_shape,
                   obs_dtype=old_obs.dtype, act_dtype=act.dtype)
    instance.store(old_obs, act, new_obs)
    return instance

  def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample obs-act-obs triples.

    Args:
        n_samples: The number of samples.

    Returns:
        old_obs: Old observations.
        act: Actions.
        new_obs: New observations.
    """
    return (self._old_obs_buffer.sample(n_samples),
            self._act_buffer.sample(n_samples),
            self._new_obs_buffer.sample(n_samples))

  def store(self, old_obs: np.ndarray, act: np.ndarray, new_obs: np.ndarray):
    """Store obs-act-obs triples.

    Args:
        old_obs: Old observations.
        act: Actions.
        new_obs: New observations.

    Raises:
        ValueError: The arguments didn't have the same length.
    """
    if not len(old_obs) == len(act) == len(new_obs):
      raise ValueError("Arguments must have the same length.")
    self._old_obs_buffer.store(old_obs)
    self._act_buffer.store(act)
    self._new_obs_buffer.store(new_obs)
    assert (len(self._old_obs_buffer)
            == len(self._act_buffer)
            == len(self._new_obs_buffer))

  def __len__(self):
    return len(self._old_obs_buffer)
