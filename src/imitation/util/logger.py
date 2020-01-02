import contextlib
import os
from typing import ContextManager, Optional, Sequence

import stable_baselines.logger as sb_logger


def _build_output_formats(folder: str,
                          format_strs: Sequence[str] = None,
                          ) -> Sequence[sb_logger.KVWriter]:
  """Build output formats for initializing a Stable Baselines Logger.

  Args:
    folder: Path to directory that logs are written to.
    format_strs: An list of output format strings. For details on available
      output formats see `stable_baselines.logger.make_output_format`.
  """
  os.makedirs(folder, exist_ok=True)
  output_formats = [sb_logger.make_output_format(f, folder)
                    for f in format_strs]
  return output_formats


class _AccumulatingLogger(sb_logger.Logger):

  def __init__(self,
               format_strs: Sequence[str],
               *,
               mean_logger: sb_logger.Logger,
               subdir: str):
    """Like Logger, except also accumulates logkv_mean on the mean_logger.

    Args:
      format_strs: A list of output format strings. For details on available
        output formats see `stable_baselines.logger.make_output_format`.
      mean_logger: A background logger that is used to keep track of log
        means.
      subdir: Used to build the logging directory. Also used in the logging
        prefix for every key written to both this logger and `mean_logger`.
    """
    self.subdir = subdir
    self.mean_logger = mean_logger
    folder = os.path.join(self.mean_logger.dir, "raw", subdir)
    output_formats = _build_output_formats(folder, format_strs)
    os.makedirs(folder, exist_ok=True)
    super().__init__(folder, output_formats)

  def logkv(self, key, val):
    raw_key = os.path.join("raw", self.subdir, key)
    super().logkv(raw_key, val)

    mean_key = os.path.join("mean", self.subdir, key)
    self.mean_logger.logkv_mean(mean_key, val)


class _HierarchicalLogger(sb_logger.Logger):

  def __init__(self,
               default_logger: sb_logger.Logger,
               format_strs: Sequence[str] = ('stdout', 'log', 'csv')):
    """A logger that forwards logging requests to one of two loggers.

    `self.current_logger` is higher priority than `default_logger` when it
    is not None. At initialization, `self.current_logger = None`. Use
    the `self.accumulate_means()` context to temporarily set
    `self.current_logger` to an `_AccumulatingLogger`.

    Args:
      default_logger: The logger to forward logging requests to when
        `self.current_logger` is None.
      format_strs: A list of output format strings that should be used by
        every `self.current_logger`. For details on available
        output formats see `stable_baselines.logger.make_output_format`.
    """
    self.default_logger = default_logger
    self.current_logger = None
    self._cached_loggers = {}
    self.format_strs = format_strs
    super().__init__(folder=self.default_logger.dir, output_formats=None)

  @contextlib.contextmanager
  def accumulate_means(self, subdir: str):
    """Temporarily use an _AccumulatingLogger as the current logger.

    Args:
        subdir: A string key for the _AccumulatingLogger which determines
          its `folder` and logging prefix. All `_AccumulatingLogger` instances are cached,
          so if this method is called again with the same `subdir` argument,
          then we load the same `_AccumulatingLogger` from last time.
    """
    if self.current_logger is not None:
      raise RuntimeError("Nested `accumulate_means` context")

    if subdir in self._cached_loggers:
      logger = self._cached_loggers[subdir]
    else:
      logger = _AccumulatingLogger(
        self.format_strs, mean_logger=self.default_logger, subdir=subdir)
      self._cached_loggers[subdir] = logger

    try:
      self.current_logger = logger
      yield
    finally:
      self.current_logger = None

  @property
  def _logger(self):
    if self.current_logger is not None:
      return self.current_logger
    else:
      return self.default_logger

  def logkv(self, key, val):
    self._logger.logkv(key, val)

  def logkv_mean(self, key, val):
    self._logger.logkv_mean(key, val)

  def dumpkvs(self):
    self._logger.dumpkvs()

  def log(self, *args, **kwargs):
    self._logger.log(*args, **kwargs)

  def get_dir(self) -> str:
    return self._logger.get_dir()

  def close(self):
    self._logger.close(self)


def _sb_logger_configure_replacement(*args, **kwargs):
  raise RuntimeError("Shouldn't call stable_baselines.logger.configure "
                     "once imitation.logger.configure() has been called")


def _sb_logger_reset_replacement():
  raise RuntimeError("Shouldn't call stable_baselines.logger.reset "
                     "once imitation.logger.configure() has been called")


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
  sb_logger.configure = _sb_logger_configure_replacement
  sb_logger.reset = _sb_logger_reset_replacement

  if format_strs is None:
    format_strs = ['stdout', 'log', 'csv']
  output_formats = _build_output_formats(folder, format_strs)
  default_logger = sb_logger.Logger(folder, output_formats)
  hier_logger = _HierarchicalLogger(default_logger, format_strs)
  sb_logger.Logger.CURRENT = hier_logger
  sb_logger.log('Logging to %s' % folder)
  assert is_configured()


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

  This context cannot be nested.

  Args:
    subdir_name: A string key for building the logger, as described above.

  Returns:
    A context manager.
  """
  assert is_configured()
  hier_logger = sb_logger.Logger.CURRENT  # type: _HierarchicalLogger
  return hier_logger.accumulate_means(subdir_name)
