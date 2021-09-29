"""Common configuration elements for reward network training."""

import logging
from typing import Any, Mapping, Optional, Type

import sacred
from stable_baselines3.common import vec_env

from imitation.rewards import reward_nets

reward_ingredient = sacred.Ingredient("reward")
logger = logging.getLogger(__name__)


@reward_ingredient.config
def config():
    # Custom reward network
    net_cls = None
    net_kwargs = {}
    locals()  # quieten flake8


@reward_ingredient.capture
def make_reward_net(
    venv: vec_env.VecEnv,
    net_cls: Optional[Type[reward_nets.RewardNet]],
    net_kwargs: Optional[Mapping[str, Any]],
) -> Optional[reward_nets.RewardNet]:
    """Builds a reward network.

    Args:
        venv: Vectorized environment reward network will predict reward for.
        net_cls: Class of reward network to construct.
        net_kwargs: Keyword arguments passed to reward network constructor.

    Returns:
        None if `reward_net_cls` is None; otherwise, an instance of `reward_net_cls`.
    """
    if net_cls is not None:
        net_kwargs = net_kwargs or {}
        reward_net = net_cls(
            venv.observation_space,
            venv.action_space,
            **net_kwargs,
        )
        logging.info(f"Reward network:\n {reward_net}")
        return reward_net
