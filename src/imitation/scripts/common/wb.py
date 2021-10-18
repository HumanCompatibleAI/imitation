"""Weights & Biases configuration elements for scripts."""

import logging
from typing import Any, Mapping, Optional

import sacred

wandb_ingredient = sacred.Ingredient("common.wandb")
logger = logging.getLogger(__name__)


@wandb_ingredient.config
def wandb_config():
    """Other user can overwrite this function to customize their wandb.init() call."""
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
