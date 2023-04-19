from __future__ import annotations
import dataclasses
import typing
from typing import Optional, Any, Mapping

if typing.TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv
    from imitation.rewards.reward_nets import RewardNet

from hydra.core.config_store import ConfigStore
from hydra.utils import call
from omegaconf import MISSING

import imitation_cli.utils.environment as environment_cg


@dataclasses.dataclass
class Config:
    _target_: str = MISSING
    environment: environment_cg.Config = MISSING


@dataclasses.dataclass
class BasicRewardNet(Config):
    _target_: str = "imitation_cli.utils.reward_network.BasicRewardNet.make"
    use_state: bool = True
    use_action: bool = True
    use_next_state: bool = False
    use_done: bool = False
    normalize_input_layer: bool = True

    @staticmethod
    def make(
            environment: VecEnv,
            normalize_input_layer: bool,
            **kwargs
    ) -> RewardNet:
        from imitation.rewards import reward_nets
        from imitation.util import networks

        reward_net = reward_nets.BasicRewardNet(
            environment.observation_space,
            environment.action_space,
            **kwargs,
        )
        if normalize_input_layer:
            reward_net = reward_nets.NormalizedRewardNet(
                reward_net,
                networks.RunningNorm,
            )
        return reward_net


@dataclasses.dataclass
class BasicShapedRewardNet(BasicRewardNet):
    _target_: str = "imitation_cli.utils.reward_network.BasicShapedRewardNet.make"
    discount_factor: float = 0.99

    @staticmethod
    def make(
            environment: VecEnv,
            normalize_input_layer: bool,
            **kwargs
    ) -> RewardNet:
        from imitation.rewards import reward_nets
        from imitation.util import networks

        reward_net = reward_nets.BasicShapedRewardNet(
            environment.observation_space,
            environment.action_space,
            **kwargs,
        )
        if normalize_input_layer:
            reward_net = reward_nets.NormalizedRewardNet(
                reward_net,
                networks.RunningNorm,
            )
        return reward_net


@dataclasses.dataclass
class RewardEnsemble(Config):
    _target_: str = "imitation_cli.utils.reward_network.RewardEnsemble.make"
    ensemble_size: int = MISSING
    ensemble_member_config: BasicRewardNet = MISSING
    add_std_alpha: Optional[float] = None

    @staticmethod
    def make(
        environment: VecEnv,
        ensemble_member_config: BasicRewardNet,
        add_std_alpha: Optional[float],
    ) -> RewardNet:
        from imitation.rewards import reward_nets

        members = [call(ensemble_member_config)]
        reward_net = reward_nets.RewardEnsemble(
            environment.observation_space, environment.action_space, members
        )
        if add_std_alpha is not None:
            reward_net = reward_nets.AddSTDRewardWrapper(
                reward_net,
                default_alpha=add_std_alpha,
            )
        return reward_net


def register_configs(group: str, defaults: Mapping[str, Any] = {}):
    cs = ConfigStore.instance()
    cs.store(group=group, name="basic", node=BasicRewardNet(**defaults))
    cs.store(group=group, name="shaped", node=BasicShapedRewardNet(**defaults))
    cs.store(group=group, name="ensemble", node=RewardEnsemble(**defaults))
