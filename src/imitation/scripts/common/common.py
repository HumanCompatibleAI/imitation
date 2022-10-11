"""Common configuration elements for scripts."""

import contextlib
import logging
import pathlib
from typing import Any, Generator, Mapping, Sequence, Tuple, Union

import numpy as np
import sacred
from stable_baselines3.common import vec_env

from imitation.data import types
from imitation.scripts.common import wb
from imitation.util import logger as imit_logger
from imitation.util import sacred as sacred_util
from imitation.util import util

common_ingredient = sacred.Ingredient("common", ingredients=[wb.wandb_ingredient])
logger = logging.getLogger(__name__)


@common_ingredient.config
def config():
    # Logging
    log_root = None
    log_dir = None
    log_level = logging.INFO
    log_format_strs = ["tensorboard", "stdout"]
    # The keys of log_format_strs_additional are concatenated to log_format_strs.
    # This allows named configs to add format strings, without changing the defaults.
    log_format_strs_additional = {}

    # Environment config
    env_name = "seals/CartPole-v0"  # environment to train on
    num_vec = 8  # number of environments in VecEnv
    parallel = True  # Use SubprocVecEnv rather than DummyVecEnv
    max_episode_steps = None  # Set to positive int to limit episode horizons
    env_make_kwargs = {}  # The kwargs passed to `spec.make`.

    locals()  # quieten flake8


@common_ingredient.config
def update_log_format_strs(log_format_strs, log_format_strs_additional):
    log_format_strs = log_format_strs + list(log_format_strs_additional.keys())


@common_ingredient.config_hook
def hook(config, command_name: str, logger):
    del logger
    updates = {}
    if config["common"]["log_dir"] is None:
        env_sanitized = config["common"]["env_name"].replace("/", "_")
        assert isinstance(env_sanitized, str)
        config_log_root = config["common"]["log_root"] or "output"
        log_root = types.parse_path(config_log_root)
        log_dir = log_root / command_name / env_sanitized / util.make_unique_timestamp()
        updates["log_dir"] = log_dir
    return updates


@common_ingredient.named_config
def wandb_logging():
    log_format_strs_additional = {"wandb": None}  # noqa: F841


@common_ingredient.named_config
def fast():
    num_vec = 2
    parallel = False  # easier to debug with everything in one process
    max_episode_steps = 5

    locals()  # quieten flake8


@common_ingredient.capture
def make_rng(_seed) -> np.random.Generator:
    """Creates a `np.random.Generator` with the given seed."""
    return np.random.default_rng(_seed)


@common_ingredient.capture
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
    parsed_log_dir = types.parse_path(log_dir)
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


@common_ingredient.capture
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


@contextlib.contextmanager
@common_ingredient.capture
def make_venv(
    env_name: str,
    num_vec: int,
    parallel: bool,
    log_dir: str,
    max_episode_steps: int,
    env_make_kwargs: Mapping[str, Any],
    **kwargs,
) -> Generator[vec_env.VecEnv, None, None]:
    """Builds the vector environment.

    Args:
        env_name: The environment to train in.
        num_vec: Number of `gym.Env` instances to combine into a vector environment.
        parallel: Whether to use "true" parallelism. If True, then use `SubProcVecEnv`.
            Otherwise, use `DummyVecEnv` which steps through environments serially.
        max_episode_steps: If not None, then a TimeLimit wrapper is applied to each
            environment to artificially limit the maximum number of timesteps in an
            episode.
        log_dir: Logs episode return statistics to a subdirectory 'monitor`.
        env_make_kwargs: The kwargs passed to `spec.make` of a gym environment.
        kwargs: Passed through to `util.make_vec_env`.

    Yields:
        The constructed vector environment.
    """
    rng = make_rng()
    # Note: we create the venv outside the try -- finally block for the case that env
    #     creation fails.
    venv = util.make_vec_env(
        env_name,
        rng=rng,
        n_envs=num_vec,
        parallel=parallel,
        max_episode_steps=max_episode_steps,
        log_dir=log_dir,
        env_make_kwargs=env_make_kwargs,
        **kwargs,
    )
    try:
        yield venv
    finally:
        venv.close()
