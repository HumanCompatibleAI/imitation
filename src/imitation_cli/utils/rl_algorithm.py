import dataclasses
import pathlib
from typing import Optional, Mapping, Any

import numpy as np
import stable_baselines3 as sb3
from hydra.utils import call
from omegaconf import MISSING

from imitation_cli.utils import environment as environment_cfg
from imitation_cli.utils import policy as policy_conf
from imitation_cli.utils import schedule


@dataclasses.dataclass
class Config:
    _target_: str = MISSING
    environment: environment_cfg.Config = MISSING


@dataclasses.dataclass
class PPO(Config):
    _target_: str = "imitation_cli.utils.rl_algorithm.PPO.make"
    # We disable recursive instantiation, so we can just make the arguments of the policy but not the policy itself
    _recursive_: bool = False
    policy: policy_conf.ActorCriticPolicy = MISSING
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
    seed: int = "${seed}"
    device: str = "auto"

    @staticmethod
    def make(
        environment: environment_cfg.Config,
        policy: policy_conf.ActorCriticPolicy,
        learning_rate: schedule.Config,
        clip_range: schedule.Config,
        **kwargs,
    ):
        policy_kwargs = policy_conf.ActorCriticPolicy.make_args(**policy)
        del policy_kwargs["use_sde"]
        del policy_kwargs["lr_schedule"]
        return sb3.PPO(
            policy=sb3.common.policies.ActorCriticPolicy,
            policy_kwargs=policy_kwargs,
            env=call(environment),
            learning_rate=call(learning_rate),
            clip_range=call(clip_range),
            **kwargs,
        )


@dataclasses.dataclass
class PPOOnDisk(PPO):
    _target_: str = "imitation_cli.utils.rl_algorithm.PPOOnDisk.make"
    path: pathlib.Path = MISSING

    @staticmethod
    def make(path: pathlib.Path, environment: environment_cfg.Config, rng: np.random.Generator):
        from imitation.policies import serialize

        return serialize.load_stable_baselines_model(sb3.PPO, path, environment)


def register_configs(group: str = "rl_algorithm", defaults: Mapping[str, Any] = {}):
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="ppo", group=group, node=PPO(**defaults))
    cs.store(name="ppo_on_disk", group=group, node=PPOOnDisk(**defaults))
