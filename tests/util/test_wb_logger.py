"""Tests `imitation.util.logger.WandbOutputFormat`."""

from typing import Any, Mapping
from unittest import mock

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

    def add(self, d):
        self._row_add(d)


class MockWandb:
    """Mock Wandb object for testing."""

    def __init__(self):
        """Initializes an instance of MockWandb."""
        self._initialized = False
        self.history = MockHistory()
        self.history_list = []
        self.history._set_callback(self._history_callback)

    def init(self, *args, **kwargs):
        self._initialized = True
        self._init_args = args
        self._init_kwargs = kwargs

    def log(
        self,
        data: Mapping[str, Any],
        step: int = None,
        commit: bool = None,
        sync: bool = None,
    ):

        assert self._initialized
        if sync:
            raise NotImplementedError("usage of sync to MockWandb.log not implemented")
        if not isinstance(data, Mapping):
            raise ValueError("`wandb.log` must be passed a dictionary")

        if any(not isinstance(key, str) for key in data.keys()):
            raise ValueError("Key values passed to `wandb.log` must be strings.")

        if step is not None:
            if self.history._step > step:
                raise ValueError(
                    "Step must only increase in log calls.  "
                    "Step {} < {}; dropping {}.".format(
                        step,
                        self.history._step,
                        data,
                    ),
                )
                return
            elif step > self.history._step:
                self.history._flush()
                self.history._step = step
        elif commit is None:
            commit = True
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


@mock.patch.object(wandb, "__init__", mock_wandb.__init__)
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
    log_obj.close()
