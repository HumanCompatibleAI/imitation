"""Config for AIRL."""
import dataclasses
from typing import Optional

from omegaconf import MISSING

from imitation_cli.utils import environment as environment_cfg
from imitation_cli.utils import (
    optimizer_class,
    reward_network,
    rl_algorithm,
    trajectories,
)


@dataclasses.dataclass
class Config:
    """Config for AIRL."""

    _target_: str = "imitation.algorithms.adversarial.airl.AIRL"
    venv: environment_cfg.Config = MISSING
    demonstrations: trajectories.Config = MISSING
    gen_algo: rl_algorithm.Config = MISSING
    reward_net: reward_network.Config = MISSING
    demo_batch_size: int = 64
    n_disc_updates_per_round: int = 2
    disc_opt_cls: optimizer_class.Config = optimizer_class.Adam()
    gen_train_timesteps: Optional[int] = None
    gen_replay_buffer_capacity: Optional[int] = None
    init_tensorboard: bool = False
    init_tensorboard_graph: bool = False
    debug_use_ground_truth: bool = False
    allow_variable_horizon: bool = True  # TODO: true just for debugging
