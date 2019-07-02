from typing import Optional, Tuple

import numpy as np


class Buffer:
  """A buffer that stores numpy arrays of fixed shape, and from which these
  arrays can be arbitrarily sampled."""

  capacity: int
  """The number of data samples that can be stored in this buffer."""

  sample_shape: Tuple[int]
  """The shape of each data sample stored in this buffer."""

  _buffer: np.ndarray
  """The underlying numpy array (which actually stores the data)."""

  _n_data: int
  """An integer in `range(0, self.capacity + 1)`. The number of
  samples currently stored in this buffer. This attribute is the return
  value of `self.__len__`."""

  _idx: int
  """An integer in `range(0, self.capacity)`.
  The index of the first row that new data should be written to."""

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
  def fromData(cls, data: np.ndarray) -> "Buffer":
    """Construct and return a Buffer containing only the provided data.

    The returned ReplayBuffer is at full capacity and ready for sampling.
    """
    capacity, *sample_shape = data.shape
    buf = cls(capacity, sample_shape, dtype=data.dtype)
    buf.store(data)
    return buf

  def store(self, data: np.ndarray) -> None:
    """Store new data samples in the buffer, replacing old samples with FIFO
    priority.

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
      # import pdb; pdb.set_trace()
      n_remain = self.capacity - self._idx
      # Need to loop around the buffer. Break into two recursive calls.
      self.store(data[:n_remain])
      assert self._idx == 0
      self.store(data[n_remain:])
    else:
      self._buffer[self._idx:new_idx] = data
      self._idx = (self._idx + len(data)) % self.capacity
      self._n_data = min(self._n_data + len(data), self.capacity)

  def sample(self, n_samples: int) -> np.ndarray:
    """Uniformly sample `n_samples` samples from the buffer without replacement.

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
    ind = np.atleast_1d(np.random.randint(len(self), size=n_samples))
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

  def __init__(self, capacity, env=None, *,
               obs_shape: Optional[Tuple[int, ...]] = None,
               act_shape: Optional[Tuple[int, ...]] = None,
               obs_dtype=None,
               act_dtype=None):
    """Constructs a ReplayBuffer.

    Args:
        env (Optional[gym.Env]): The environment whose action and observation
            spaces can be used to determine the data shapes of the underlying
            buffers. Overrides the `obs_shape` and `act_shape` arguments.
        obs_shape (Optional[Tuple[int, ...]]): The shape of the observation
            space.
        act_shape (Optional[Tuple[int, ...]]): The shape of the action
            space.
        obs_dtype (Optional[dtype]): The dtype of the observation space.
        action_dtype (Optional[dtype]): The dtype of the action space.
    Raises:
        ValueError: Couldn't infer the observation and action shapes and dtypes
            from the arguments.
    """
    if env is not None:
      obs_shape = tuple(env.observation_space.shape)
      act_shape = tuple(env.action_space.shape)
      obs_dtype = env.observation_space.dtype
      act_dtype = env.action_space.dtype
    if obs_shape is None or act_shape is None:
      raise ValueError("Couldn't infer both the observation and action shapes "
                       "from the arguments.")

    self.capacity = capacity
    self._old_obs_buffer = Buffer(capacity, obs_shape, obs_dtype)
    self._act_buffer = Buffer(capacity, act_shape, act_dtype)
    self._new_obs_buffer = Buffer(capacity, obs_shape, obs_dtype)

  @classmethod
  def fromData(cls, old_obs: np.ndarray, act: np.ndarray, new_obs: np.ndarray
               ) -> "ReplayBuffer":
    """Construct and return a ReplayBuffer containing only the provided data.

    The returned ReplayBuffer is at full capacity and ready for sampling.

    Args:
      old_obs: Old observations.
      act: Actions.
      new_obs: New observations.
    Returns:
      (ReplayBuffer): A new ReplayBuffer.
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
    assert len(self._old_obs_buffer) == len(self._act_buffer) \
        == len(self._new_obs_buffer)

  def __len__(self):
    return len(self._old_obs_buffer)
