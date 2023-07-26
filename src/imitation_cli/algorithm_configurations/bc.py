from __future__ import annotations
import dataclasses

import typing
from typing import Optional

from imitation_cli.algorithm_configurations import base

if typing.TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv

from omegaconf import MISSING

from imitation_cli.utils import environment as environment_cfg, randomness, trajectories, optimizer_class
from imitation_cli.utils import policy as policy_cfg


@dataclasses.dataclass
class Config(base.DemonstrationAlgorithmConfig):
    """Config for BC."""
    _target_: str = "imitation_cli.algorithm_configurations.bc.Config.make"
    venv: environment_cfg.Config = MISSING
    rng: randomness.Config = MISSING
    policy: Optional[policy_cfg.ActorCriticPolicy] = MISSING
    batch_size: int = 32
    minibatch_size: Optional[int] = None
    optimizer_cls: optimizer_class.Config = optimizer_class.Adam
    optimizer_kwargs: Optional[dict] = None
    ent_weight: float = 1e-3
    l2_weight: float = 0.0
    device: str = "auto"

    @staticmethod
    def make(venv: VecEnv, **kwargs):
        from imitation.algorithms import bc
        return bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            **kwargs,
        )


@dataclasses.dataclass
class TrainConfig:
    """Config for BC training."""
    n_epochs: Optional[int] = 2  #TODO: find proper default
    n_batches: Optional[int] = None  #TODO: find proper default


