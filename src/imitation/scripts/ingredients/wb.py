"""This ingredient provides Weights & Biases logging."""

import logging
from typing import Any, Mapping, Optional

import sacred

wandb_ingredient = sacred.Ingredient("logging.wandb")
logger = logging.getLogger(__name__)


@wandb_ingredient.config
def wandb_config():
    # Other users can overwrite this function to customize their wandb.init() call.
    wandb_tag = None  # User-specified tag for this run
    wandb_name_prefix = ""  # User-specified prefix for the run name
    wandb_kwargs = dict(
        project="imitation",
        monitor_gym=False,
        save_code=False,
    )  # Other kwargs to pass to wandb.init()
    wandb_additional_info = dict()

    locals()


@wandb_ingredient.capture
def wandb_init(
    _run,
    wandb_name_prefix: str,
    wandb_tag: Optional[str],
    wandb_kwargs: Mapping[str, Any],
    wandb_additional_info: Mapping[str, Any],
    log_dir: str,
) -> None:
    """Putting everything together to get the W&B kwargs for wandb.init().

    Args:
        wandb_name_prefix: User-specified prefix for wandb run name.
        wandb_tag: User-specified tag for this run.
        wandb_kwargs: User-specified kwargs for wandb.init().
        wandb_additional_info: User-specific additional info to add to wandb experiment
            ``config``.
        log_dir: W&B logs will be stored in directory `{log_dir}/wandb/`.

    Raises:
        ModuleNotFoundError: wandb is not installed.
    """
    env_name = _run.config["environment"]["gym_id"]
    root_seed = _run.config["seed"]

    updated_wandb_kwargs: Mapping[str, Any] = {
        **wandb_kwargs,
        "name": f"{wandb_name_prefix}-{env_name}-seed{root_seed}",
        "tags": [env_name, f"seed{root_seed}"] + ([wandb_tag] if wandb_tag else []),
        "dir": log_dir,
    }
    try:
        import wandb
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Trying to call `wandb.init()` but `wandb` not installed: "
            "try `pip install wandb`.",
        ) from e
    wandb_config_dict = dict(**_run.config)
    wandb_config_dict.update(wandb_additional_info)
    wandb.init(config=wandb_config_dict, **updated_wandb_kwargs)
