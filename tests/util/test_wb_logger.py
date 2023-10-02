"""Tests `imitation.util.logger.WandbOutputFormat`."""

import sys
from typing import Any, Mapping, Optional
from unittest import mock

import pytest
import wandb

from imitation.util import logger


class MockHistory:
    """Mock History object for testing."""

    def __init__(self):
        """Initializes an instance of MockHistory."""
        self._step = 0
        self._data = dict()
        self._callback = None

    def _set_callback(self, cb):
        self._callback = cb

    def _row_update(self, row):
        self._data.update(row)

    def _row_add(self, row):
        self._data.update(row)
        self._flush()
        self._step += 1

    def _flush(self):
        if len(self._data) > 0:
            self._data["_step"] = self._step
            if self._callback:
                self._callback(row=self._data, step=self._step)
            self._data = dict()


class MockWandb:
    """Mock Wandb object for testing."""

    def __init__(self):
        """Initializes an instance of MockWandb."""
        self._initialized = False
        self.history = MockHistory()
        self.history_list = []
        self.history._set_callback(self._history_callback)
        self._init_args = None
        self._init_kwargs = None

    def init(self, *args, **kwargs):
        self._initialized = True
        self._init_args = args
        self._init_kwargs = kwargs

    def log(
        self,
        data: Mapping[str, Any],
        step: Optional[int] = None,
        commit: bool = False,
        sync: bool = False,
    ):
        assert self._initialized
        if sync:
            raise NotImplementedError("usage of sync to MockWandb.log not implemented")

        if step is not None:
            if step > self.history._step:
                self.history._flush()
                self.history._step = step
        if commit:
            self.history._row_add(data)
        else:
            self.history._row_update(data)

    def _history_callback(self, row: Mapping[str, Any], step: int) -> None:
        self.history_list.append(row)

    def finish(self):
        assert self._initialized
        self._initialized = False


mock_wandb = MockWandb()


# we ignore the type below as one should technically not access the
# __init__ method directly but only by creating an instance.
@mock.patch.object(wandb, "__init__", mock_wandb.__init__)  # type: ignore[misc]
@mock.patch.object(wandb, "init", mock_wandb.init)
@mock.patch.object(wandb, "log", mock_wandb.log)
@mock.patch.object(wandb, "finish", mock_wandb.finish)
def test_wandb_output_format():
    wandb.init()
    log_obj = logger.configure(format_strs=["wandb"])
    assert len(mock_wandb.history_list) == 0, "nothing should be logged yet"
    log_obj.info("test 123")
    assert len(mock_wandb.history_list) == 0, "nothing should be logged yet"
    log_obj.record("foo", 42)
    assert len(mock_wandb.history_list) == 0, "nothing should be logged yet"
    log_obj.record("fow", 24, exclude="wandb")
    log_obj.record("fizz", 12, exclude="stdout")
    log_obj.dump()
    assert len(mock_wandb.history_list) == 1, "exactly one entry should be logged"
    assert mock_wandb.history_list == [{"_step": 0, "foo": 42, "fizz": 12}]
    log_obj.record("fizz", 21)
    log_obj.dump(step=3)
    assert len(mock_wandb.history_list) == 2, "exactly two entries should be logged"
    assert mock_wandb.history_list == [
        {"_step": 0, "foo": 42, "fizz": 12},
        {"_step": 3, "fizz": 21},
    ]
    log_obj.close()


def test_wandb_module_import_error():
    wandb_module = sys.modules["wandb"]
    try:
        sys.modules["wandb"] = None
        with pytest.raises(ModuleNotFoundError, match=r"Trying to log data.*"):
            logger.configure(format_strs=["wandb"])
    finally:
        sys.modules[wandb] = wandb_module
