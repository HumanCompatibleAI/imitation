import numpy as np
import pytest

from imitation.util.buffer import Buffer, ReplayBuffer


@pytest.mark.parametrize("capacity", [10, 30, 60])
@pytest.mark.parametrize("chunk_len", [1, 2, 4, 9])
@pytest.mark.parametrize("sample_shape", [(), (1, 2), (5, 4, 4)])
def test_buffer(capacity, chunk_len, sample_shape):
  """Builds a Buffer with the provided `capacity` and insert `capacity * 3`
  samples into the buffer in chunks of shape `(chunk_len,) + sample_shape`.

  We always insert the same chunk, an array containing only 66.6.

  * `len(buffer)` should increase until we reach capacity.
  * `buffer._idx` should loop between 0 and `capacity - 1`.
  * After every insertion, samples should only contain 66.6.
  * Mutating the inserted chunk shouldn't mutate the buffer.
  """
  buf = Buffer(capacity, sample_shape=sample_shape, dtype=float)
  data = np.full((chunk_len,) + sample_shape, 66.6)

  for i in range(0, capacity*3, chunk_len):
    assert len(buf) == min(i, capacity)
    assert buf._idx == i % capacity
    buf.store(data)
    samples = buf.sample(100)
    assert samples.shape == (100,) + sample_shape
    assert np.all(samples == 66.6)

  # Confirm that buffer is not mutable from inserted sample.
  data[:] = np.nan
  samples = buf.sample(100)
  assert samples.shape == (100,) + sample_shape
  assert np.all(samples == 66.6)


@pytest.mark.parametrize("capacity", [30, 60])
@pytest.mark.parametrize("chunk_len", [1, 4, 9])
@pytest.mark.parametrize("obs_shape", [(), (1, 2)])
@pytest.mark.parametrize("act_shape", [(), (5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.int, np.bool, np.float32])
def test_replay_buffer(capacity, chunk_len, obs_shape, act_shape, dtype):
  """Builds a ReplayBuffer with the provided `capacity` and inserts
  `capacity * 3` observation-action-observation samples into the buffer in
  chunks of length `chunk_len`.

  All chunks are of the appopriate observation or action shape, and contain
  the value fill_val.

  `len(buffer)` should increase until we reach capacity.
  `buffer._idx` should loop between 0 and `capacity - 1`.
  After every insertion, samples should only contain 66.6.
  """
  buf = ReplayBuffer(capacity, obs_shape=obs_shape, act_shape=act_shape,
                     obs_dtype=dtype, act_dtype=dtype)
  old_obs_fill_val, act_fill_val, new_obs_fill_val = (
      np.array(x, dtype=dtype) for x in [0, 1, 2])
  old_obs_data = np.full(
      (chunk_len,) + obs_shape, old_obs_fill_val, dtype=dtype)
  act_data = np.full((chunk_len,) + act_shape, act_fill_val, dtype=dtype)
  new_obs_data = np.full(
      (chunk_len,) + obs_shape, new_obs_fill_val, dtype=dtype)

  for i in range(0, capacity*3, chunk_len):
    assert len(buf) == min(i, capacity)
    for b in [buf._old_obs_buffer, buf._act_buffer, buf._new_obs_buffer]:
      assert b._idx == i % capacity

    buf.store(old_obs_data, act_data, new_obs_data)

    old_obs, acts, new_obs = buf.sample(100)
    assert old_obs.shape == new_obs.shape == (100,) + obs_shape
    assert acts.shape == (100,) + act_shape

    assert np.all(old_obs == old_obs_fill_val)
    assert np.all(acts == act_fill_val)
    assert np.all(new_obs == new_obs_fill_val)

    assert old_obs.dtype == dtype
    assert acts.dtype == dtype
    assert new_obs.dtype == dtype


@pytest.mark.parametrize("sample_shape", [(), (1,), (5, 2)])
def test_buffer_store_errors(sample_shape):
  capacity = 11
  dtype = "float32"

  def buf():
      return Buffer(capacity, sample_shape, dtype)

  # `data` is empty.
  b = buf()
  with pytest.raises(ValueError):
      b.store(np.empty((0,) + sample_shape, dtype=dtype))

  # `data` has too many samples.
  b = buf()
  with pytest.raises(ValueError):
      b.store(np.empty((capacity + 1,) + sample_shape, dtype=dtype))

  # `data` has the wrong sample shape.
  b = buf()
  with pytest.raises(ValueError):
      b.store(np.empty((1, 3, 3, 3, 3), dtype=dtype))


def test_buffer_sample_errors():
  b = Buffer(10, (2, 1), dtype=bool)
  with pytest.raises(ValueError):
      b.sample(5)


def test_replay_buffer_init_errors():
  with pytest.raises(ValueError, match=r"Specified.* and environment"):
    ReplayBuffer(15, env="MockEnv", obs_shape=(10, 10))
  with pytest.raises(ValueError, match=r"Shape or dtype missing.*"):
      ReplayBuffer(15, obs_shape=(10, 10), act_shape=(15,), obs_dtype=bool)
  with pytest.raises(ValueError, match=r"Shape or dtype missing.*"):
      ReplayBuffer(15, obs_shape=(10, 10), obs_dtype=bool, act_dtype=bool)


def test_replay_buffer_store_errors():
  b = ReplayBuffer(10, obs_shape=(), obs_dtype=bool, act_shape=(),
                   act_dtype=float)
  with pytest.raises(ValueError, match=".* same length.*"):
      b.store(np.ones(4), np.ones(4), np.ones(3))


def test_buffer_from_data():
  data = np.ndarray([50, 30], dtype=bool)
  buf = Buffer.from_data(data)
  assert buf._buffer is not data
  assert data.dtype == buf._buffer.dtype
  assert np.array_equal(buf._buffer, data)


def test_replay_buffer_from_data():
  old_obs = np.array([5, 2], dtype=int)
  act = np.ones((2, 6), dtype=float)
  new_obs = np.array([7, 8], dtype=int)
  buf = ReplayBuffer.from_data(old_obs, act, new_obs)
  assert np.array_equal(buf._old_obs_buffer._buffer, old_obs)
  assert np.array_equal(buf._new_obs_buffer._buffer, new_obs)
  assert np.array_equal(buf._act_buffer._buffer, act)

  with pytest.raises(ValueError, match=r".*same length."):
    new_obs_toolong = np.array([7, 8, 9], dtype=int)
    ReplayBuffer.from_data(old_obs, act, new_obs_toolong)
  with pytest.raises(ValueError, match=r".*same dtype."):
    new_obs_float = np.array(new_obs, dtype=float)
    ReplayBuffer.from_data(old_obs, act, new_obs_float)
