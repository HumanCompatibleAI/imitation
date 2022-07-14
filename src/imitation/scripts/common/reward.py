"""Common configuration elements for reward network training."""

import logging
from typing import Any, Mapping, Optional, Type

import sacred
from stable_baselines3.common import vec_env
from torch import nn

from imitation.rewards import reward_nets
from imitation.util import networks

reward_ingredient = sacred.Ingredient("reward")
logger = logging.getLogger(__name__)


@reward_ingredient.config
def config():
    # Custom reward network
    net_cls = None
    add_std_alpha = None
    net_kwargs = {}
    normalize_output_layer = networks.RunningNorm
    locals()  # quieten flake8


@reward_ingredient.named_config
def normalize_input_disable():
    net_kwargs = {"normalize_input_layer": None}  # noqa: F841


@reward_ingredient.named_config
def normalize_input_running():
    net_kwargs = {"normalize_input_layer": networks.RunningNorm}  # noqa: F841


@reward_ingredient.named_config
def normalize_output_disable():
    normalize_output_layer = None  # noqa: F841


@reward_ingredient.named_config
def normalize_output_running():
    normalize_output_layer = networks.RunningNorm  # noqa: F841


@reward_ingredient.named_config
def reward_ensemble():
    net_cls = reward_nets.RewardEnsemble  # noqa: F841
    add_std_alpha = 0  # noqa: F841


@reward_ingredient.config_hook
def config_hook(config, command_name, logger):
    """Sets default values for `net_cls` and `net_kwargs`."""
    del logger
    res = {}
    if config["reward"]["net_cls"] is None:
        default_net = reward_nets.BasicRewardNet
        if command_name == "airl":
            default_net = reward_nets.BasicShapedRewardNet
        res["net_cls"] = default_net
    if "normalize_input_layer" not in config["reward"]["net_kwargs"]:
        res["net_kwargs"] = {"normalize_input_layer": networks.RunningNorm}
    return res


@reward_ingredient.capture
def make_reward_net(
    venv: vec_env.VecEnv,
    net_cls: Type[reward_nets.RewardNet],
    net_kwargs: Mapping[str, Any],
    normalize_output_layer: Optional[Type[nn.Module]],
    add_std_alpha: Optional[float],
) -> reward_nets.RewardNet:
    """Builds a reward network.

    Args:
        venv: Vectorized environment reward network will predict reward for.
        net_cls: Class of reward network to construct.
        net_kwargs: Keyword arguments passed to reward network constructor.
        normalize_output_layer: Wrapping the reward_net with NormalizedRewardNet
            to normalize the reward output.
        add_std_alpha: multiple of reward function standard deviation to add to the
            reward in predict_processed. Must be None when using a reward function that
            does not keep track of variance. Defaults to None.

    Returns:
        A, possibly wrapped, instance of `net_cls`.
    """
    reward_net = reward_nets.make_reward_net(
        venv.observation_space,
        venv.action_space,
        net_cls,
        net_kwargs,
        normalize_output_layer,
        add_std_alpha,
    )
    logging.info(f"Reward network:\n {reward_net}")
    return reward_net
