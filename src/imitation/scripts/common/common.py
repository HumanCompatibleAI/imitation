"""Common configuration elements for scripts."""

import logging
import os
from typing import Any, Mapping, Sequence, Tuple, Union, Optional

import sacred
from stable_baselines3.common import vec_env

from imitation.util import logger as imit_logger
from imitation.util import sacred as sacred_util
from imitation.util import util

common_ingredient = sacred.Ingredient("common")
logger = logging.getLogger(__name__)


@common_ingredient.config
def config():
    # Logging
    log_root = None
    log_dir = None
    log_level = logging.INFO
    log_format_strs = ["tensorboard", "stdout"]

    # Environment config
    env_name = "seals/CartPole-v0"  # environment to train on
    num_vec = 8  # number of environments in VecEnv
    parallel = True  # Use SubprocVecEnv rather than DummyVecEnv
    max_episode_steps = None  # Set to positive int to limit episode horizons
    env_make_kwargs = {}  # The kwargs passed to `spec.make`.

    # Capture root-seed for Sacred, this is the seed specified from the command line
    # It is different from _seed
    root_seed = None

    locals()  # quieten flake8


@common_ingredient.config
def wandb_config(env_name, root_seed):
    wandb_logging = False  # If True, adding a custom writer that logs to wandb 
    wandb_tag = None  # User-specified tag for this run
    wandb_name_suffix = ""
    assert isinstance(root_seed, int), "common.root_seed must be specified for wandb"
    wandb_kwargs = dict(
        project="imitation",
        name="-".join([env_name, f"seed{root_seed}",]) + wandb_name_suffix,
        tags=[env_name, f"seed{root_seed}"] + ([wandb_tag] if wandb_tag else []),
        monitor_gym=False,
        save_code=False,
    )


@common_ingredient.config_hook
def hook(config, command_name, logger):
    del logger
    updates = {}
    if config["common"]["log_dir"] is None:
        env_sanitized = config["common"]["env_name"].replace("/", "_")
        log_dir = os.path.join(
            "output",
            command_name,
            env_sanitized,
            util.make_unique_timestamp(),
        )
        updates["log_dir"] = log_dir
    return updates


@common_ingredient.named_config
def fast():
    num_vec = 2
    parallel = False  # easier to debug with everything in one process
    max_episode_steps = 5

    locals()  # quieten flake8


@common_ingredient.capture
def make_log_dir(
    _run,
    log_dir: str,
    log_level: Union[int, str],
) -> str:
    """Creates log directory and sets up symlink to Sacred logs.

    Args:
        log_dir: The directory to log to.
        log_level: The threshold of the logger. Either an integer level (10, 20, ...),
            a string of digits ('10', '20'), or a string of the designated level
            ('DEBUG', 'INFO', ...).

    Returns:
        The `log_dir`. This avoids the caller needing to capture this argument.
    """
    os.makedirs(log_dir, exist_ok=True)
    # convert strings of digits to numbers; but leave levels like 'INFO' unmodified
    try:
        log_level = int(log_level)
    except ValueError:
        pass
    logging.basicConfig(level=log_level)
    logger.info("Logging to %s", log_dir)
    sacred_util.build_sacred_symlink(log_dir, _run)
    return log_dir


@common_ingredient.capture
def make_wandb_kwargs(
    log_dir: str, 
    wandb_kwargs: Mapping[str, Any],
) -> Mapping[str, Any]:
    """W&B kwargs for wandb.init(). Other user can overwrite this function to
    customize their wandb.init() call.

    Args:
        # env_name (str): Environment name.
        # seed (Optional[int]): User-specified root-seed from command line.
        # wandb_tag (Optional[str]): User-sepcified tag for this run.
        log_dir (str): W&B logs will be stored into directory `{log_dir}/wandb/`.

    Returns:
        Mapping: kwargs for wandb.init()
    """
    updated_wandb_kwargs = {}
    updated_wandb_kwargs.update(wandb_kwargs)
    updated_wandb_kwargs.update(dict(dir=log_dir,))
    return updated_wandb_kwargs


@common_ingredient.capture
def setup_logging(
    _run,
    wandb_logging: bool,
    log_format_strs: Sequence[str],
) -> Tuple[imit_logger.HierarchicalLogger, str]:
    """Builds the imitation logger.

    Args:
        log_format_strs: The types of formats to log to.

    Returns:
        The configured imitation logger and `log_dir`.
        Returning `log_dir` avoids the caller needing to capture this value.
    """
    log_dir = make_log_dir()
    
    wandb_writer = None
    if wandb_logging:
        wandb_kwargs = make_wandb_kwargs()
        wandb_writer = imit_logger.WandbOutputFormat(
            wandb_kwargs=wandb_kwargs,
            config=_run.config,
        )

    custom_logger = imit_logger.configure(
            folder=os.path.join(log_dir, "log"), 
            format_strs=log_format_strs, 
            custom_writers=[wandb_writer] if wandb_writer else None
        )
    return custom_logger, log_dir


@common_ingredient.capture
def make_venv(
    _seed,
    env_name: str,
    num_vec: int,
    parallel: bool,
    log_dir: str,
    max_episode_steps: int,
    env_make_kwargs: Mapping[str, Any],
    **kwargs,
) -> vec_env.VecEnv:
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

    Returns:
        The constructed vector environment.
    """
    return util.make_vec_env(
        env_name,
        num_vec,
        seed=_seed,
        parallel=parallel,
        max_episode_steps=max_episode_steps,
        log_dir=log_dir,
        env_make_kwargs=env_make_kwargs,
        **kwargs,
    )
