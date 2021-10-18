"""Tests for `imitation.util.registry`."""

import pytest

from imitation.util import registry


def test_lazy():
    """Test indirect/lazy loading of registered values."""
    reg = registry.Registry()

    reg.register("nomodule", indirect="this.module.does.not.exist:foobar")
    with pytest.raises(ImportError):
        reg.get("nomodule")

    reg.register("noattribute", indirect="imitation:attr_does_not_exist")
    with pytest.raises(AttributeError):
        reg.get("noattribute")

    with pytest.raises(ValueError, match="exactly one of"):
        reg.register(key="wrongargs", value=3.14, indirect="math:pi")

    reg.register("exists", indirect="math:pi")
    val = reg.get("exists")
    import math

    assert val == math.pi


def test_keys():
    reg = registry.Registry()

    with pytest.raises(KeyError, match="not registered"):
        reg.get("foobar")

    reg.register(key="foobar", value="fizzbuzz")
    assert reg.get("foobar") == "fizzbuzz"

    with pytest.raises(KeyError, match="Duplicate registration"):
        reg.register(key="foobar", value="fizzbuzz")
