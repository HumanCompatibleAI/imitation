"""Utilities for seeding random number generators."""
from __future__ import annotations

import dataclasses
import typing

from omegaconf import MISSING

if typing.TYPE_CHECKING:
    import numpy as np


@dataclasses.dataclass
class Config:
    """Configuration for seeding random number generators."""

    _target_: str = "imitation_cli.utils.randomness.Config.make"
    seed: int = MISSING

    @staticmethod
    def make(seed: int) -> np.random.Generator:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)

        return np.random.default_rng(seed)
