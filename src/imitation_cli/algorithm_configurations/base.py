import dataclasses

from omegaconf import MISSING

from imitation_cli.utils import trajectories


@dataclasses.dataclass
class BaseImitationAlgorithmConfig:
    # TODO: add _logger field go generate a HierarchicalLogger
    allow_variable_horizon: bool = False

    
@dataclasses.dataclass
class DemonstrationAlgorithmConfig(BaseImitationAlgorithmConfig):
    """Config for running a demonstration-based imitation learning algorithm."""
    demonstrations: trajectories.Config = MISSING
