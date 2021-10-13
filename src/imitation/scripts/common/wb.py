"""Weights & Biases configuration elements for scripts."""

import logging
from typing import Any, Mapping, Optional

import sacred
import stable_baselines3.common.logger as sb_logger

from imitation.util import logger as imit_logger

wandb_ingredient = sacred.Ingredient("wandb")
logger = logging.getLogger(__name__)


@wandb_ingredient.config
def wandb_config():
    """Other user can overwrite this function to customize their wandb.init() call."""
    wandb_logging = False  # If True, adding a custom writer that logs to wandb
    wandb_tag = None  # User-specified tag for this run
    wandb_name_suffix = ""  # User-specified suffix for the run name
    wandb_kwargs = dict(
        project="imitation",
        monitor_gym=False,
        save_code=False,
    )  # Other kwargs to pass to wandb.init()

    locals()


@wandb_ingredient.capture
def make_wandb_kwargs(
    _run,
    wandb_name_suffix: str,
    wandb_tag: Optional[str],
    wandb_kwargs: Mapping[str, Any],
    log_dir: str,
) -> Mapping[str, Any]:
    """Putting everything together to get the W&B kwargs for wandb.init().

    Args:
        wandb_name_suffix (str): User-specified suffix for wandb run name.
        wandb_tag (Optional[str]): User-sepcified tag for this run.
        wandb_kwargs (Mapping[str, Any]): User-specified kwargs for wandb.init().
        log_dir (str): W&B logs will be stored into directory `{log_dir}/wandb/`.

    Returns:
        Mapping: kwargs for wandb.init()
    """
    env_name = _run.config["common"]["env_name"]
    root_seed = _run.config["seed"]

    updated_wandb_kwargs = {}
    updated_wandb_kwargs.update(wandb_kwargs)
    updated_wandb_kwargs.update(
        dict(
            name="-".join([env_name, f"seed{root_seed}"]) + wandb_name_suffix,
            tags=[env_name, f"seed{root_seed}"] + ([wandb_tag] if wandb_tag else []),
            dir=log_dir,
        ),
    )
    return updated_wandb_kwargs


@wandb_ingredient.capture
def setup_wandb_writer(
    _run,
    wandb_logging: bool,
    custom_logger: sb_logger.Logger,
    log_dir: str,
):
    """Capture function to add a wandb writer to the custom logger."""
    wandb_writer = None
    if wandb_logging:
        wandb_kwargs = make_wandb_kwargs(log_dir=log_dir)
        wandb_writer = imit_logger.WandbOutputFormat(
            wandb_kwargs=wandb_kwargs,
            config=_run.config,
        )
    custom_logger.output_formats += [wandb_writer]
    return custom_logger
