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
@pytest.mark.parametrize("fill_val,dtype", [
  (32, np.int), (True, np.bool), (32.4, np.float32)])
def test_replay_buffer(capacity, chunk_len, obs_shape, act_shape, fill_val,
                       dtype):
  """Builds a ReplayBuffer with the provided `capacity` and insert
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
  act_data = np.full((chunk_len,) + act_shape, fill_val)
  obs_data = np.full((chunk_len,) + obs_shape, fill_val)

  for i in range(0, capacity*3, chunk_len):
    assert len(buf) == min(i, capacity)
    for b in [buf._old_obs_buffer, buf._act_buffer, buf._new_obs_buffer]:
      assert b._idx == i % capacity

    buf.store(obs_data, act_data, obs_data)

    old_obs, acts, new_obs = buf.sample(100)
    assert old_obs.shape == new_obs.shape == (100,) + obs_shape
    assert acts.shape == (100,) + act_shape

    assert np.all(old_obs == fill_val)
    assert np.all(acts == fill_val)
    assert np.all(new_obs == fill_val)

    assert old_obs.dtype == dtype
    assert acts.dtype == dtype
    assert new_obs.dtype == dtype
