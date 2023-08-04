"""This ingredient provides a number of logging utilities.

It is responsible for logging to WandB, TensorBoard, and stdout.
It will also create a symlink to the sacred logging directory in the log directory.
"""

import logging
import pathlib
from typing import Sequence, Tuple, Union

import huggingface_sb3 as hfsb3
import sacred

from imitation.scripts.ingredients import environment, wb
from imitation.util import logger as imit_logger
from imitation.util import sacred as sacred_util
from imitation.util import util

logging_ingredient = sacred.Ingredient(
    "logging",
    ingredients=[
        wb.wandb_ingredient,
        environment.environment_ingredient,
    ],
)
logger = logging.getLogger(__name__)


@logging_ingredient.config
def config():
    # Logging
    log_root = None
    log_dir = None
    log_level = logging.INFO
    log_format_strs = ["tensorboard", "stdout"]
    # The keys of log_format_strs_additional are concatenated to log_format_strs.
    # This allows named configs to add format strings, without changing the defaults.
    log_format_strs_additional = {}

    locals()  # silence flake8 unused variable warning


@logging_ingredient.config
def update_log_format_strs(log_format_strs, log_format_strs_additional):
    log_format_strs = log_format_strs + list(log_format_strs_additional.keys())


@logging_ingredient.config_hook
def hook(config, command_name: str, logger):
    del logger
    updates = {}
    if config["logging"]["log_dir"] is None:
        config_log_root = config["logging"]["log_root"] or "output"
        log_root = util.parse_path(config_log_root)
        env_sanitized = hfsb3.EnvironmentName(config["environment"]["gym_id"])
        assert isinstance(env_sanitized, str)
        log_dir = log_root / command_name / env_sanitized / util.make_unique_timestamp()
        updates["log_dir"] = log_dir
    return updates


@logging_ingredient.named_config
def wandb_logging():
    log_format_strs_additional = {"wandb": None}  # noqa: F841


@logging_ingredient.capture
def make_log_dir(
    _run,
    log_dir: str,
    log_level: Union[int, str],
) -> pathlib.Path:
    """Creates log directory and sets up symlink to Sacred logs.

    Args:
        log_dir: The directory to log to.
        log_level: The threshold of the logger. Either an integer level (10, 20, ...),
            a string of digits ('10', '20'), or a string of the designated level
            ('DEBUG', 'INFO', ...).

    Returns:
        The `log_dir`. This avoids the caller needing to capture this argument.
    """
    parsed_log_dir = util.parse_path(log_dir)
    parsed_log_dir.mkdir(parents=True, exist_ok=True)
    # convert strings of digits to numbers; but leave levels like 'INFO' unmodified
    try:
        log_level = int(log_level)
    except ValueError:
        pass
    logging.basicConfig(level=log_level)
    logger.info("Logging to %s", parsed_log_dir)
    sacred_util.build_sacred_symlink(parsed_log_dir, _run)
    return parsed_log_dir


@logging_ingredient.capture
def setup_logging(
    _run,
    log_format_strs: Sequence[str],
) -> Tuple[imit_logger.HierarchicalLogger, pathlib.Path]:
    """Builds the imitation logger.

    Args:
        log_format_strs: The types of formats to log to.

    Returns:
        The configured imitation logger and `log_dir`.
        Returning `log_dir` avoids the caller needing to capture this value.
    """
    log_dir = make_log_dir()
    if "wandb" in log_format_strs:
        wb.wandb_init(log_dir=str(log_dir))
    custom_logger = imit_logger.configure(
        folder=log_dir / "log",
        format_strs=log_format_strs,
    )
    return custom_logger, log_dir
