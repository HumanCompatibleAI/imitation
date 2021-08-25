import pytest

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
