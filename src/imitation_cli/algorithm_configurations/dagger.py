import dataclasses
import pathlib
from typing import Optional

from omegaconf import MISSING

import imitation_cli.utils.environment as environment_cfg
from imitation_cli.algorithm_configurations import bc, base
from imitation_cli.utils import randomness, trajectories, schedule, policy


@dataclasses.dataclass
class Config(base.BaseImitationAlgorithmConfig):
    """Config for DAgger."""
    _target_: str = "imitation.algorithms.dagger.SimpleDAggerTrainer"
    venv: environment_cfg.Config = MISSING
    scratch_dir: pathlib.Path = MISSING
    expert_policy: policy.Config = MISSING
    rng: randomness.Config = MISSING
    expert_trajs: Optional[trajectories.Config] = None
    beta_schedule: Optional[schedule.Config] = None
    bc_trainer: bc.Config = MISSING
