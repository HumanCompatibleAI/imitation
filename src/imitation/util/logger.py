import contextlib
import datetime
import logging
import os
import tempfile
from typing import Optional, Sequence

import stable_baselines3.common.logger as sb_logger

from imitation.data import types


def _build_output_formats(
    folder: str,
    format_strs: Sequence[str] = None,
) -> Sequence[sb_logger.KVWriter]:
    """Build output formats for initializing a Stable Baselines Logger.

    Args:
      folder: Path to directory that logs are written to.
      format_strs: An list of output format strings. For details on available
        output formats see `stable_baselines3.logger.make_output_format`.
    """
    os.makedirs(folder, exist_ok=True)
    output_formats = [sb_logger.make_output_format(f, folder) for f in format_strs]
    return output_formats


class HierarchicalLogger(sb_logger.Logger):
    def __init__(
        self,
        default_logger: sb_logger.Logger,
        format_strs: Sequence[str] = ("stdout", "log", "csv"),
    ):
        """A logger with a context for accumulating mean values.

        Args:
          default_logger: The default logger when not in the a `accumulate_means`
            context. Also the logger to which mean values are written to when
            contexts are over.
          format_strs: An list of output format strings that should be used by
            every Logger initialized by this class during an `AccumulatingMeans`
            context. For details on available output formats see
            `stable_baselines3.logger.make_output_format`.
        """
        self.default_logger = default_logger
        self.current_logger = None
        self._cached_loggers = {}
        self._subdir = None
        self.format_strs = format_strs
        super().__init__(folder=self.default_logger.dir, output_formats=[])

    @contextlib.contextmanager
    def accumulate_means(self, subdir: types.AnyPath):
        """Temporarily modifies this HierarchicalLogger to accumulate means values.

        During this context, `self.record(key, value)` writes the "raw" values in
        "{self.default_logger.log_dir}/{subdir}" under the key "raw/{subdir}/{key}".
        At the same time, any call to `self.record` will also accumulate mean values
        on the default logger by calling
        `self.default_logger.record_mean(f"mean/{subdir}/{key}", value)`.

        During the context, `self.record(key, value)` will write the "raw" values in
        `"{self.default_logger.log_dir}/subdir"` under the key "raw/{subdir}/key".

        After the context exits, calling `self.dump()` will write the means
        of all the "raw" values accumulated during this context to
        `self.default_logger` under keys with the prefix `mean/{subdir}/`

        Note that the behavior of other logging methods, `log` and `record_mean`
        are unmodified and will go straight to the default logger.

        Args:
          subdir: A string key which determines the `folder` where raw data is
            written and temporary logging prefixes for raw and mean data. Entering
            an `accumulate_means` context in the future with the same `subdir`
            will safely append to logs written in this folder rather than
            overwrite.
        """
        if self.current_logger is not None:
            raise RuntimeError("Nested `accumulate_means` context")

        if subdir in self._cached_loggers:
            logger = self._cached_loggers[subdir]
        else:
            subdir = types.path_to_str(subdir)
            folder = os.path.join(self.default_logger.dir, "raw", subdir)
            os.makedirs(folder, exist_ok=True)
            output_formats = _build_output_formats(folder, self.format_strs)
            logger = sb_logger.Logger(folder, output_formats)
            self._cached_loggers[subdir] = logger

        try:
            self.current_logger = logger
            self._subdir = subdir
            yield
        finally:
            self.current_logger = None
            self._subdir = None

    def record(self, key, val, exclude=None):
        if self.current_logger is not None:  # In accumulate_means context.
            assert self._subdir is not None
            raw_key = os.path.join("raw", self._subdir, key)
            self.current_logger.record(raw_key, val, exclude)

            mean_key = os.path.join("mean", self._subdir, key)
            self.default_logger.record_mean(mean_key, val, exclude)
        else:  # Not in accumulate_means context.
            self.default_logger.record(key, val, exclude)

    @property
    def _logger(self):
        if self.current_logger is not None:
            return self.current_logger
        else:
            return self.default_logger

    def dump(self, step=0):
        self._logger.dump(step)

    def get_dir(self) -> str:
        return self._logger.get_dir()

    def log(self, *args, **kwargs):
        self.default_logger.log(*args, **kwargs)

    def set_level(self, level: int) -> None:
        self.default_logger.set_level(level)

    def record_mean(self, key, val, exclude=None):
        self.default_logger.record_mean(key, val, exclude)

    def close(self):
        self.default_logger.close()
        for logger in self._cached_loggers.values():
            logger.close()


def configure(
    folder: Optional[types.AnyPath] = None, format_strs: Optional[Sequence[str]] = None
) -> HierarchicalLogger:
    """Configure Stable Baselines logger to be `accumulate_means()`-compatible.

    After this function is called, `stable_baselines3.logger.{configure,reset}()`
    are replaced with stubs that raise RuntimeError.

    Args:
        folder: Argument from `stable_baselines3.logger.configure`.
        format_strs: An list of output format strings. For details on available
          output formats see `stable_baselines3.logger.make_output_format`.
    """
    if folder is None:
        now = datetime.datetime.now()
        timestamp = now.strftime("imitation-%Y-%m-%d-%H-%M-%S-%f")
        folder = os.path.join(tempfile.gettempdir(), timestamp)
    else:
        folder = types.path_to_str(folder)
    logging.info("Logging to '%s'", folder)
    if format_strs is None:
        format_strs = ["stdout", "log", "csv"]
    output_formats = _build_output_formats(folder, format_strs)
    default_logger = sb_logger.Logger(folder, output_formats)
    hier_logger = HierarchicalLogger(default_logger, format_strs)
    return hier_logger
