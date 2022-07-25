"""Tests `imitation.util.sacred` and `imitation.util.util`."""

import warnings

import numpy as np
import pytest
import torch as th
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from imitation.util import sacred as sacred_util
from imitation.util import util


def test_endless_iter():
    x = range(2)
    it = util.endless_iter(x)
    assert next(it) == 0
    assert next(it) == 1
    assert next(it) == 0


def test_endless_iter_error():
    x = []
    with pytest.raises(ValueError, match="no elements"):
        util.endless_iter(x)


@given(
    arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=10),
        elements=st.floats(min_value=1e-3, max_value=1e6, allow_nan=False),
    ).map(
        # Compute the fractional part of the sum of the elements, divide it by
        # the number of elements, and subtract this from every element.
        # This ensures that the sum of the elements is integral.
        lambda x: (x - (x.sum() - np.floor(x.sum())) / len(x)),
    ),
)
def test_integer_constrained_rounding(x: np.ndarray):
    original_sum = x.sum()

    rounded = util.oric(x)
    assert np.allclose(rounded.sum(), original_sum)
    assert np.abs(x - rounded).max() <= 1.0


def test_dict_get_nested():
    assert sacred_util.dict_get_nested({}, "asdf.foo", default=4) == 4
    assert sacred_util.dict_get_nested({"a": {"b": "c"}}, "a.b") == "c"


def test_safe_to_tensor():
    # Copy iff the array is non-writable.
    numpy = np.array([1, 2, 3])
    torch = util.safe_to_tensor(numpy)
    assert np.may_share_memory(numpy, torch)

    # This should never cause a warning from `th.as_tensor`.
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        numpy.flags.writeable = False
        torch = util.safe_to_tensor(numpy)
        assert not np.may_share_memory(numpy, torch)


def test_tensor_iter_norm():
    # vector is [1,0,1,1,-5,-6]; its 2-norm is 8, and 1-norm is 14
    tensor_list = [
        th.tensor([1.0, 0.0]),
        th.tensor([[1.0], [1.0], [-5.0]]),
        th.tensor([-6.0]),
    ]
    norm_2 = util.tensor_iter_norm(tensor_list, ord=2).item()
    assert np.allclose(norm_2, 8.0)
    norm_1 = util.tensor_iter_norm(tensor_list, ord=1).item()
    assert np.allclose(norm_1, 14.0)
    with pytest.raises(ValueError):
        util.tensor_iter_norm(tensor_list, ord=0.0)
