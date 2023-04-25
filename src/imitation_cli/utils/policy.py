"""Configurable policies for SB3 Base Policies.""" ""
from __future__ import annotations

import dataclasses
import pathlib
import typing
from typing import Any, Dict, List, Optional, Union, cast

if typing.TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv
    from stable_baselines3.common.policies import BasePolicy

from hydra.core.config_store import ConfigStore
from hydra.utils import call
from omegaconf import MISSING

from imitation_cli.utils import activation_function_class as act_fun_class_cfg
from imitation_cli.utils import environment as environment_cfg
from imitation_cli.utils import feature_extractor_class as feature_extractor_class_cfg
from imitation_cli.utils import optimizer_class as optimizer_class_cfg
from imitation_cli.utils import schedule


@dataclasses.dataclass
class Config:
    """Base configuration for policies."""

    _target_: str = MISSING
    environment: environment_cfg.Config = MISSING


@dataclasses.dataclass
class Random(Config):
    """Configuration for a random policy."""

    _target_: str = "imitation_cli.utils.policy.Random.make"

    @staticmethod
    def make(environment: VecEnv) -> BasePolicy:
        from imitation.policies import base

        return base.RandomPolicy(
            environment.observation_space,
            environment.action_space,
        )


@dataclasses.dataclass
class ZeroPolicy(Config):
    """Configuration for a zero policy."""

    _target_: str = "imitation_cli.utils.policy.ZeroPolicy.make"

    @staticmethod
    def make(environment: VecEnv) -> BasePolicy:
        from imitation.policies import base

        return base.ZeroPolicy(environment.observation_space, environment.action_space)


@dataclasses.dataclass
class ActorCriticPolicy(Config):
    """Configuration for a stable-baselines3 ActorCriticPolicy."""

    _target_: str = "imitation_cli.utils.policy.ActorCriticPolicy.make"
    lr_schedule: schedule.Config = schedule.FixedSchedule(3e-4)
    net_arch: Optional[Dict[str, List[int]]] = None
    activation_fn: act_fun_class_cfg.Config = act_fun_class_cfg.TanH()
    ortho_init: bool = True
    use_sde: bool = False
    log_std_init: float = 0.0
    full_std: bool = True
    use_expln: bool = False
    squash_output: bool = False
    features_extractor_class: feature_extractor_class_cfg.Config = (
        feature_extractor_class_cfg.FlattenExtractorConfig()
    )
    features_extractor_kwargs: Optional[Dict[str, Any]] = None
    share_features_extractor: bool = True
    normalize_images: bool = True
    optimizer_class: optimizer_class_cfg.Config = optimizer_class_cfg.Adam()
    optimizer_kwargs: Optional[Dict[str, Any]] = None

    @staticmethod
    def make_args(
        activation_fn: act_fun_class_cfg.Config,
        features_extractor_class: feature_extractor_class_cfg.Config,
        optimizer_class: optimizer_class_cfg.Config,
        **kwargs,
    ):
        del kwargs["_target_"]
        del kwargs["environment"]

        kwargs["activation_fn"] = call(activation_fn)
        kwargs["features_extractor_class"] = call(features_extractor_class)
        kwargs["optimizer_class"] = call(optimizer_class)

        return dict(
            **kwargs,
        )

    @staticmethod
    def make(
        environment: VecEnv,
        **kwargs,
    ) -> BasePolicy:
        import stable_baselines3 as sb3

        return sb3.common.policies.ActorCriticPolicy(
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            **kwargs,
        )


@dataclasses.dataclass
class Loaded(Config):
    """Base configuration for a policy that is loaded from somewhere."""

    policy_type: str = (
        "PPO"  # The SB3 policy class. Only SAC and PPO supported as of now
    )

    @staticmethod
    def type_to_class(policy_type: str):
        import stable_baselines3 as sb3

        policy_type = policy_type.lower()
        if policy_type == "ppo":
            return sb3.PPO
        if policy_type == "ppo":
            return sb3.SAC
        raise ValueError(f"Unknown policy type {policy_type}")


@dataclasses.dataclass
class PolicyOnDisk(Loaded):
    """Configuration for a policy that is loaded from a path on disk."""

    _target_: str = "imitation_cli.utils.policy.PolicyOnDisk.make"
    path: pathlib.Path = MISSING

    @staticmethod
    def make(
        environment: VecEnv,
        policy_type: str,
        path: pathlib.Path,
    ) -> BasePolicy:
        from imitation.policies import serialize

        return serialize.load_stable_baselines_model(
            Loaded.type_to_class(policy_type),
            str(path),
            environment,
        ).policy


@dataclasses.dataclass
class PolicyFromHuggingface(Loaded):
    """Configuration for a policy that is loaded from a HuggingFace model."""

    _target_: str = "imitation_cli.utils.policy.PolicyFromHuggingface.make"
    _recursive_: bool = False
    organization: str = "HumanCompatibleAI"

    @staticmethod
    def make(
        environment: environment_cfg.Config,
        policy_type: str,
        organization: str,
    ) -> BasePolicy:
        import huggingface_sb3 as hfsb3

        from imitation.policies import serialize

        model_name = hfsb3.ModelName(
            policy_type.lower(),
            hfsb3.EnvironmentName(environment.env_name),
        )
        repo_id = hfsb3.ModelRepoId(organization, model_name)
        filename = hfsb3.load_from_hub(repo_id, model_name.filename)
        model = serialize.load_stable_baselines_model(
            Loaded.type_to_class(policy_type),
            filename,
            call(environment),
        )
        return model.policy


def register_configs(
    group: str,
    default_environment: Optional[Union[environment_cfg.Config, str]] = MISSING,
):
    default_environment = cast(environment_cfg.Config, default_environment)
    cs = ConfigStore.instance()
    cs.store(group=group, name="random", node=Random(environment=default_environment))
    cs.store(group=group, name="zero", node=ZeroPolicy(environment=default_environment))
    cs.store(
        group=group,
        name="on_disk",
        node=PolicyOnDisk(environment=default_environment),
    )
    cs.store(
        group=group,
        name="from_huggingface",
        node=PolicyFromHuggingface(environment=default_environment),
    )
    cs.store(
        group=group,
        name="actor_critic",
        node=ActorCriticPolicy(environment=default_environment),
    )
    schedule.register_configs(group=group + "/lr_schedule")
