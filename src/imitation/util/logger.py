import contextlib
import os
from typing import ContextManager, Optional, Sequence

import stable_baselines.logger as sb_logger


def _build_output_formats(
    folder: str, format_strs: Sequence[str] = None,
) -> Sequence[sb_logger.KVWriter]:
    """Build output formats for initializing a Stable Baselines Logger.

    Args:
      folder: Path to directory that logs are written to.
      format_strs: An list of output format strings. For details on available
        output formats see `stable_baselines.logger.make_output_format`.
    """
    os.makedirs(folder, exist_ok=True)
    output_formats = [sb_logger.make_output_format(f, folder) for f in format_strs]
    return output_formats


class _HierarchicalLogger(sb_logger.Logger):
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
            `stable_baselines.logger.make_output_format`.
        """
        self.default_logger = default_logger
        self.current_logger = None
        self._cached_loggers = {}
        self._subdir = None
        self.format_strs = format_strs
        super().__init__(folder=self.default_logger.dir, output_formats=None)

    @contextlib.contextmanager
    def accumulate_means(self, subdir: str):
        """Temporarily modifies this _HierarchicalLogger to accumulate means values.

        During this context, `self.logkv(key, value)` writes the "raw" values in
        "{self.default_logger.log_dir}/{subdir}" under the key "raw/{subdir}/{key}".
        At the same time, any call to `self.logkv` will also accumulate mean values
        on the default logger by calling
        `self.default_logger.logkv_mean(f"mean/{subdir}/{key}", value)`.

        During the context, `self.logkv(key, value)` will write the "raw" values in
        `"{self.default_logger.log_dir}/subdir"` under the key "raw/{subdir}/key".

        After the context exits, calling `self.dumpkvs()` will write the means
        of all the "raw" values accumulated during this context to
        `self.default_logger` under keys with the prefix `mean/{subdir}/`

        Note that the behavior of other logging methods, `log` and `logkv_mean`
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
            folder = os.path.join(self.default_logger.dir, "raw", subdir)
            os.makedirs(folder)
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

    def logkv(self, key, val):
        if self.current_logger is not None:
            assert self._subdir is not None
            raw_key = os.path.join("raw", self._subdir, key)
            self.current_logger.logkv(raw_key, val)

            mean_key = os.path.join("mean", self._subdir, key)
            self.default_logger.logkv_mean(mean_key, val)
        else:
            self.default_logger.logkv_mean(key, val)

    @property
    def _logger(self):
        if self.current_logger is not None:
            return self.current_logger
        else:
            return self.default_logger

    def dumpkvs(self):
        self._logger.dumpkvs()

    def get_dir(self) -> str:
        return self._logger.get_dir()

    def log(self, *args, **kwargs):
        self.default_logger.log(*args, **kwargs)

    def logkv_mean(self, key, val):
        self.default_logger.logkv_mean(key, val)

    def close(self):
        raise NotImplementedError


def _sb_logger_configure_replacement(*args, **kwargs):
    raise RuntimeError(
        "Shouldn't call stable_baselines.logger.configure "
        "once imitation.logger.configure() has been called"
    )


def _sb_logger_reset_replacement():
    raise RuntimeError(
        "Shouldn't call stable_baselines.logger.reset "
        "once imitation.logger.configure() has been called"
    )


def is_configured() -> bool:
    """Return True if the custom logger is active."""
    return isinstance(sb_logger.Logger.CURRENT, _HierarchicalLogger)


def configure(folder: str, format_strs: Optional[Sequence[str]] = None) -> None:
    """Configure Stable Baselines logger to be `accumulate_means()`-compatible.

    After this function is called, `stable_baselines.logger.{configure,reset}()`
    are replaced with stubs that raise RuntimeError.

    Args:
        folder: Argument from `stable_baselines.logger.configure`.
        format_strs: An list of output format strings. For details on available
          output formats see `stable_baselines.logger.make_output_format`.
    """
    # Replace `stable_baselines.logger` methods with erroring stubs to
    # prevent unexpected logging state from mixed logging configuration.
    sb_logger.configure = _sb_logger_configure_replacement
    sb_logger.reset = _sb_logger_reset_replacement

    if format_strs is None:
        format_strs = ["stdout", "log", "csv"]
    output_formats = _build_output_formats(folder, format_strs)
    default_logger = sb_logger.Logger(folder, output_formats)
    hier_logger = _HierarchicalLogger(default_logger, format_strs)
    sb_logger.Logger.CURRENT = hier_logger
    sb_logger.log("Logging to %s" % folder)
    assert is_configured()


def logkv(key, val) -> None:
    """Alias for `stable_baselines.logger.logkv`."""
    sb_logger.logkv(key, val)


def dumpkvs() -> None:
    """Alias for `stable_baselines.logger.dumpkvs`."""
    sb_logger.dumpkvs()


def accumulate_means(subdir_name: str) -> ContextManager:
    """Temporarily redirect logkv() to a different logger and auto-track kvmeans.

    Within this context, the original logger is swapped out for a special logger
    in directory `"{current_logging_dir}/raw/{subdir_name}"`.

    The special logger's `stable_baselines.logger.logkv(key, val)`, in addition
    to tracking its own logs, also forwards the log to the original logger's
    `.logkv_mean()` under the key `mean/{subdir_name}/{key}`.

    After the context exits, these means can be dumped as usual using
    `stable_baselines.logger.dumpkvs()` or `imitation.util.logger.dumpkvs()`.

    Note that the behavior of other logging methods, `log` and `logkv_mean`
    are unmodified and will go straight to the original logger.

    This context cannot be nested.

    Args:
      subdir_name: A string key for building the logger, as described above.

    Returns:
      A context manager.
    """
    assert is_configured()
    hier_logger = sb_logger.Logger.CURRENT  # type: _HierarchicalLogger
    return hier_logger.accumulate_means(subdir_name)
