import numpy as np
import pytest

from imitation.util.buffer import Buffer, ReplayBuffer


def _fill_chunk(start, chunk_len, sample_shape, dtype=np.float):
  fill_vals = np.arange(start, start + chunk_len, dtype=dtype)
  fill_vals = np.reshape(fill_vals, (-1, ) + (1,) * len(sample_shape))
  chunk = np.tile(fill_vals, (1,) + sample_shape)
  return chunk


def _get_fill_from_chunk(chunk):
  chunk_len, *sample_shape = chunk.shape
  sample_size = max(1, np.prod(sample_shape))
  return chunk.flatten()[::sample_size]


def _check_bound(end, capacity, samples, offset=0):
  start = max(0, end - capacity)
  assert np.all(start + offset <= samples), "samples violate lower bound"
  assert np.all(samples <= end + offset), "samples violate upper bound"


@pytest.mark.parametrize("capacity", [10, 30, 60])
@pytest.mark.parametrize("chunk_len", [1, 2, 4, 9])
@pytest.mark.parametrize("sample_shape", [(), (1, 2), (5, 4, 4)])
def test_buffer(capacity, chunk_len, sample_shape):
  """Builds a Buffer with the provided `capacity` and insert `capacity * 3`
  samples into the buffer in chunks of shape `(chunk_len,) + sample_shape`.
  We always insert chunks with consecutive integers.

  * `len(buffer)` should increase until we reach capacity.
  * `buffer._idx` should loop between 0 and `capacity - 1`.
  * After every insertion, samples should be in expected range, verifying
    FIFO insertion.
  * Mutating the inserted chunk shouldn't mutate the buffer.
  """
  buf = Buffer(capacity,
               sample_shapes={'a': sample_shape, 'b': sample_shape},
               dtypes={'a': float, 'b': float})

  for i in range(0, capacity*3, chunk_len):
    assert len(buf) == min(i, capacity)
    assert buf._idx == i % capacity
    chunk = _fill_chunk(i, chunk_len, sample_shape)
    buf.store({'a': chunk, 'b': chunk})
    for samples in buf.sample(100).values():
      assert samples.shape == (100,) + sample_shape
      _check_bound(i + chunk_len, capacity, samples)

    # Confirm that buffer is not mutable from inserted sample.
    chunk[:] = np.nan
    assert not np.any(np.isnan(buf._arrays['a']))
    assert not np.any(np.isnan(buf._arrays['b']))


@pytest.mark.parametrize("capacity", [30, 60])
@pytest.mark.parametrize("chunk_len", [1, 4, 9])
@pytest.mark.parametrize("obs_shape", [(), (1, 2)])
@pytest.mark.parametrize("act_shape", [(), (5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.int, np.float32])
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

  for i in range(0, capacity*3, chunk_len):
    assert len(buf) == min(i, capacity)
    assert buf._buffer._idx == i % capacity

    old_obs_data = _fill_chunk(i, chunk_len, obs_shape, dtype=dtype)
    new_obs_data = _fill_chunk(3 * capacity + i, chunk_len, obs_shape,
                               dtype=dtype)
    act_data = _fill_chunk(6 * capacity + i, chunk_len, act_shape, dtype=dtype)

    buf.store(old_obs_data, act_data, new_obs_data)

    # Are samples right shape?
    old_obs, acts, new_obs = buf.sample(100)
    assert old_obs.shape == new_obs.shape == (100,) + obs_shape
    assert acts.shape == (100,) + act_shape

    # Are samples right data type?
    assert old_obs.dtype == dtype
    assert acts.dtype == dtype
    assert new_obs.dtype == dtype

    # Are samples in range?
    _check_bound(i + chunk_len, capacity, old_obs)
    _check_bound(i + chunk_len, capacity, new_obs, 3 * capacity)
    _check_bound(i + chunk_len, capacity, acts, 6 * capacity)

    # Are samples in-order?
    old_obs_fill = _get_fill_from_chunk(old_obs)
    new_obs_fill = _get_fill_from_chunk(new_obs)
    act_fill = _get_fill_from_chunk(acts)

    assert np.all(new_obs_fill - old_obs_fill == 3 * capacity), "out of order"
    assert np.all(act_fill - new_obs_fill == 3 * capacity), "out of order"


@pytest.mark.parametrize("sample_shape", [(), (1,), (5, 2)])
def test_buffer_store_errors(sample_shape):
  capacity = 11
  dtype = "float32"

  def buf():
    return Buffer(capacity, {'k': sample_shape}, {'k': dtype})

  # Unexpected keys
  b = buf()
  with pytest.raises(ValueError):
    b.store({})
  chunk = np.ones((1, ) + sample_shape)
  with pytest.raises(ValueError):
    b.store({'y': chunk})
  with pytest.raises(ValueError):
    b.store({'k': chunk, 'y': chunk})

  # `data` is empty.
  b = buf()
  with pytest.raises(ValueError):
    b.store({'k': np.empty((0,) + sample_shape, dtype=dtype)})

  # `data` has too many samples.
  b = buf()
  with pytest.raises(ValueError):
    b.store({'k': np.empty((capacity + 1,) + sample_shape, dtype=dtype)})

  # `data` has the wrong sample shape.
  b = buf()
  with pytest.raises(ValueError):
    b.store({'k': np.empty((1, 3, 3, 3, 3), dtype=dtype)})


def test_buffer_sample_errors():
  b = Buffer(10, {'k': (2, 1)}, dtypes={'k': bool})
  with pytest.raises(ValueError):
    b.sample(5)


def test_buffer_init_errors():
  with pytest.raises(KeyError, match=r"sample_shape and dtypes.*"):
    Buffer(10, dict(a=(2, 1), b=(3,)), dtypes=dict(a="float32", c=bool))


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
  buf = Buffer.from_data({'k': data})
  assert buf._arrays['k'] is not data
  assert data.dtype == buf._arrays['k'].dtype
  assert np.array_equal(buf._arrays['k'], data)


def test_replay_buffer_from_data():
  old_obs = np.array([5, 2], dtype=int)
  act = np.ones((2, 6), dtype=float)
  new_obs = np.array([7, 8], dtype=int)
  buf = ReplayBuffer.from_data(old_obs, act, new_obs)
  assert np.array_equal(buf._buffer._arrays['old_obs'], old_obs)
  assert np.array_equal(buf._buffer._arrays['new_obs'], new_obs)
  assert np.array_equal(buf._buffer._arrays['act'], act)

  with pytest.raises(ValueError, match=r".*same length."):
    new_obs_toolong = np.array([7, 8, 9], dtype=int)
    ReplayBuffer.from_data(old_obs, act, new_obs_toolong)
  with pytest.raises(ValueError, match=r".*same dtype."):
    new_obs_float = np.array(new_obs, dtype=float)
    ReplayBuffer.from_data(old_obs, act, new_obs_float)
