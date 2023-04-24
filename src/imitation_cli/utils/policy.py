from __future__ import annotations
import dataclasses
import pathlib
import typing
from typing import Any, Dict, List, Optional, Mapping

if typing.TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv
    from stable_baselines3.common.policies import BasePolicy

from hydra.core.config_store import ConfigStore
from hydra.utils import call
from omegaconf import MISSING

from imitation_cli.utils import \
    activation_function_class as activation_function_class_cfg, \
    environment as environment_cfg,\
    feature_extractor_class as feature_extractor_class_cfg,\
    optimizer_class as optimizer_class_cfg, \
    schedule


@dataclasses.dataclass
class Config:
    _target_: str = MISSING
    environment: environment_cfg.Config = MISSING


@dataclasses.dataclass
class Random(Config):
    _target_: str = "imitation_cli.utils.policy.Random.make"

    @staticmethod
    def make(environment: VecEnv) -> BasePolicy:
        from imitation.policies import base
        return base.RandomPolicy(environment.observation_space, environment.action_space)


@dataclasses.dataclass
class ZeroPolicy(Config):
    _target_: str = "imitation_cli.utils.policy.ZeroPolicy.make"

    @staticmethod
    def make(environment: VecEnv) -> BasePolicy:
        from imitation.policies import base

        return base.ZeroPolicy(environment.observation_space, environment.action_space)


@dataclasses.dataclass
class ActorCriticPolicy(Config):
    _target_: str = "imitation_cli.utils.policy.ActorCriticPolicy.make"
    lr_schedule: schedule.Config = schedule.FixedSchedule(3e-4)  # TODO: make sure this is copied from the rl_algorithm instead
    net_arch: Optional[Dict[str, List[int]]] = None
    activation_fn: activation_function_class_cfg.Config = activation_function_class_cfg.TanH()
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
        activation_fn: activation_function_class_cfg.Config,
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
    type: str = "PPO"  # The SB3 policy class. Only SAC and PPO supported as of now

    @staticmethod
    def type_to_class(type: str):
        import stable_baselines3 as sb3

        type = type.lower()
        if type == "ppo":
            return sb3.PPO
        if type == "ppo":
            return sb3.SAC
        raise ValueError(f"Unknown policy type {type}")


@dataclasses.dataclass
class PolicyOnDisk(Loaded):
    _target_: str = "imitation_cli.utils.policy.PolicyOnDisk.make"
    path: pathlib.Path = MISSING

    @staticmethod
    def make(
        environment: VecEnv,
        type: str,
        path: pathlib.Path,
    ) -> BasePolicy:
        from imitation.policies import serialize

        return serialize.load_stable_baselines_model(
            Loaded.type_to_class(type), str(path), environment
        ).policy


@dataclasses.dataclass
class PolicyFromHuggingface(Loaded):
    _target_: str = "imitation_cli.utils.policy.PolicyFromHuggingface.make"
    organization: str = "HumanCompatibleAI"

    @staticmethod
    def make(
        environment: VecEnv,
        type: str,
        organization: str,
    ) -> BasePolicy:
        import huggingface_sb3 as hfsb3

        from imitation.policies import serialize

        model_name = hfsb3.ModelName(
            type.lower(), hfsb3.EnvironmentName(environment.gym_id)
        )
        repo_id = hfsb3.ModelRepoId(organization, model_name)
        filename = hfsb3.load_from_hub(repo_id, model_name.filename)
        model = serialize.load_stable_baselines_model(
            Loaded.type_to_class(type), filename, environment
        )
        return model.policy


def register_configs(group: str):
    cs = ConfigStore.instance()
    cs.store(group=group, name="random", node=Random)
    cs.store(group=group, name="zero", node=ZeroPolicy)
    cs.store(group=group, name="on_disk", node=PolicyOnDisk)
    cs.store(group=group, name="from_huggingface", node=PolicyFromHuggingface)
    cs.store(group=group, name="actor_critic", node=ActorCriticPolicy)
