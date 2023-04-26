"""Configurable RL algorithms."""
from __future__ import annotations

import dataclasses
import pathlib
import typing
from typing import Optional, Union, cast

if typing.TYPE_CHECKING:
    import stable_baselines3 as sb3
    from stable_baselines3.common.vec_env import VecEnv

from hydra.utils import instantiate, to_absolute_path
from omegaconf import MISSING

from imitation_cli.utils import environment as environment_cfg
from imitation_cli.utils import policy as policy_cfg
from imitation_cli.utils import schedule


@dataclasses.dataclass
class Config:
    """Base configuration for RL algorithms."""

    _target_: str = MISSING
    environment: environment_cfg.Config = MISSING


@dataclasses.dataclass
class PPO(Config):
    """Configuration for a stable-baselines3 PPO algorithm."""

    _target_: str = "imitation_cli.utils.rl_algorithm.PPO.make"
    # We disable recursive instantiation, so we can just make the
    # arguments of the policy but not the policy itself
    _recursive_: bool = False
    policy: policy_cfg.ActorCriticPolicy = policy_cfg.ActorCriticPolicy()
    learning_rate: schedule.Config = schedule.FixedSchedule(3e-4)
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: schedule.Config = schedule.FixedSchedule(0.2)
    clip_range_vf: Optional[schedule.Config] = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: Optional[float] = None
    tensorboard_log: Optional[str] = None
    verbose: int = 0
    seed: int = MISSING
    device: str = "auto"

    @staticmethod
    def make(
        environment: environment_cfg.Config,
        policy: policy_cfg.ActorCriticPolicy,
        learning_rate: schedule.Config,
        clip_range: schedule.Config,
        **kwargs,
    ) -> sb3.PPO:
        import stable_baselines3 as sb3

        policy_kwargs = policy_cfg.ActorCriticPolicy.make_args(
            **typing.cast(dict, policy),
        )
        del policy_kwargs["use_sde"]
        del policy_kwargs["lr_schedule"]
        return sb3.PPO(
            policy=sb3.common.policies.ActorCriticPolicy,
            policy_kwargs=policy_kwargs,
            env=instantiate(environment),
            learning_rate=instantiate(learning_rate),
            clip_range=instantiate(clip_range),
            **kwargs,
        )


@dataclasses.dataclass
class PPOOnDisk(Config):
    """Configuration for a stable-baselines3 PPO algorithm loaded from disk."""

    _target_: str = "imitation_cli.utils.rl_algorithm.PPOOnDisk.make"
    path: pathlib.Path = MISSING

    @staticmethod
    def make(environment: VecEnv, path: pathlib.Path) -> sb3.PPO:
        import stable_baselines3 as sb3

        from imitation.policies import serialize

        return serialize.load_stable_baselines_model(
            sb3.PPO,
            str(to_absolute_path(str(path))),
            environment,
        )


def register_configs(
    group: str = "rl_algorithm",
    default_environment: Optional[Union[environment_cfg.Config, str]] = MISSING,
    default_seed: Optional[Union[int, str]] = MISSING,
):
    from hydra.core.config_store import ConfigStore

    default_environment = cast(environment_cfg.Config, default_environment)
    default_seed = cast(int, default_seed)

    cs = ConfigStore.instance()
    cs.store(
        name="ppo",
        group=group,
        node=PPO(
            environment=default_environment,
            policy=policy_cfg.ActorCriticPolicy(environment=default_environment),
            seed=default_seed,
        ),
    )
    cs.store(
        name="ppo_on_disk",
        group=group,
        node=PPOOnDisk(environment=default_environment),
    )

    schedule.register_configs(group=group + "/learning_rate")
    schedule.register_configs(group=group + "/clip_range")
