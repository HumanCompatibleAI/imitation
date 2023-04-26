"""Reward network configuration."""
from __future__ import annotations

import dataclasses
import typing
from typing import Optional, Union, cast

if typing.TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv
    from imitation.rewards.reward_nets import RewardNet

from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING

import imitation_cli.utils.environment as environment_cfg


@dataclasses.dataclass
class Config:
    """Base configuration for reward networks."""

    _target_: str = MISSING
    environment: environment_cfg.Config = MISSING


@dataclasses.dataclass
class BasicRewardNet(Config):
    """Configuration for a basic reward network."""

    _target_: str = "imitation_cli.utils.reward_network.BasicRewardNet.make"
    use_state: bool = True
    use_action: bool = True
    use_next_state: bool = False
    use_done: bool = False
    normalize_input_layer: bool = True

    @staticmethod
    def make(environment: VecEnv, normalize_input_layer: bool, **kwargs) -> RewardNet:
        from imitation.rewards import reward_nets
        from imitation.util import networks

        reward_net = reward_nets.BasicRewardNet(
            environment.observation_space,
            environment.action_space,
            **kwargs,
        )
        if normalize_input_layer:
            return reward_nets.NormalizedRewardNet(
                reward_net,
                networks.RunningNorm,
            )
        else:
            return reward_net


@dataclasses.dataclass
class BasicShapedRewardNet(BasicRewardNet):
    """Configuration for a basic shaped reward network."""

    _target_: str = "imitation_cli.utils.reward_network.BasicShapedRewardNet.make"
    discount_factor: float = 0.99

    @staticmethod
    def make(environment: VecEnv, normalize_input_layer: bool, **kwargs) -> RewardNet:
        from imitation.rewards import reward_nets
        from imitation.util import networks

        reward_net = reward_nets.BasicShapedRewardNet(
            environment.observation_space,
            environment.action_space,
            **kwargs,
        )
        if normalize_input_layer:
            return reward_nets.NormalizedRewardNet(
                reward_net,
                networks.RunningNorm,
            )
        else:
            return reward_net


@dataclasses.dataclass
class RewardEnsemble(Config):
    """Configuration for a reward ensemble."""

    _target_: str = "imitation_cli.utils.reward_network.RewardEnsemble.make"
    _recursive_: bool = False
    ensemble_size: int = MISSING
    ensemble_member_config: BasicRewardNet = MISSING
    add_std_alpha: Optional[float] = None

    @staticmethod
    def make(
        environment: environment_cfg.Config,
        ensemble_member_config: BasicRewardNet,
        add_std_alpha: Optional[float],
        ensemble_size: int,
    ) -> RewardNet:
        from imitation.rewards import reward_nets

        venv = instantiate(environment)
        reward_net = reward_nets.RewardEnsemble(
            venv.observation_space,
            venv.action_space,
            [instantiate(ensemble_member_config) for _ in range(ensemble_size)],
        )
        if add_std_alpha is not None:
            return reward_nets.AddSTDRewardWrapper(
                reward_net,
                default_alpha=add_std_alpha,
            )
        else:
            return reward_net


def register_configs(
    group: str,
    default_environment: Optional[Union[environment_cfg.Config, str]] = MISSING,
):
    default_environment = cast(environment_cfg.Config, default_environment)
    cs = ConfigStore.instance()
    cs.store(
        group=group,
        name="basic",
        node=BasicRewardNet(environment=default_environment),
    )
    cs.store(
        group=group,
        name="shaped",
        node=BasicShapedRewardNet(environment=default_environment),
    )
    cs.store(
        group=group,
        name="small_ensemble",
        node=RewardEnsemble(
            environment=default_environment,
            ensemble_size=5,
            ensemble_member_config=BasicRewardNet(environment=default_environment),
        ),
    )
