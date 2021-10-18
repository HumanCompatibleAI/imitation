"""Tests `imitation.util.sacred` and `imitation.util.util`."""

import numpy as np
import pytest
import torch as th

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


def test_dict_get_nested():
    assert sacred_util.dict_get_nested({}, "asdf.foo", default=4) == 4
    assert sacred_util.dict_get_nested({"a": {"b": "c"}}, "a.b") == "c"


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
