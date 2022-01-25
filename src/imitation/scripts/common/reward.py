"""Common configuration elements for reward network training."""

import logging
from typing import Any, Mapping, Type

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
    net_kwargs = {}
    locals()  # quieten flake8


@reward_ingredient.named_config
def normalize_disable():
    net_kwargs = {"normalize_layer": None}  # noqa: F841


@reward_ingredient.named_config
def normalize_batchnorm():
    net_kwargs = {"normalize_layer": nn.BatchNorm1d}  # noqa: F841


@reward_ingredient.named_config
def normalize_running():
    net_kwargs = {"normalize_layer": networks.RunningNorm}  # noqa: F841


@reward_ingredient.config_hook
def config_hook(config, command_name, logger):
    del logger
    res = {}
    if config["reward"]["net_cls"] is None:
        default_net = reward_nets.BasicRewardNet
        if command_name == "airl":
            default_net = reward_nets.BasicShapedRewardNet
        res["net_cls"] = default_net
    if "normalize_layer" not in config["reward"]["net_kwargs"]:
        res["net_kwargs"] = {"normalize_layer": networks.RunningNorm}
    return res


@reward_ingredient.capture
def make_reward_net(
    venv: vec_env.VecEnv,
    net_cls: Type[reward_nets.RewardNet],
    net_kwargs: Mapping[str, Any],
) -> reward_nets.RewardNet:
    """Builds a reward network.

    Args:
        venv: Vectorized environment reward network will predict reward for.
        net_cls: Class of reward network to construct.
        net_kwargs: Keyword arguments passed to reward network constructor.

    Returns:
        None if `reward_net_cls` is None; otherwise, an instance of `reward_net_cls`.
    """
    reward_net = net_cls(
        venv.observation_space,
        venv.action_space,
        **net_kwargs,
    )
    logging.info(f"Reward network:\n {reward_net}")
    return reward_net
