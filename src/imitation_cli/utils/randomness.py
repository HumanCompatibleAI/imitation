from __future__ import annotations
import dataclasses
import typing

if typing.TYPE_CHECKING:
    import numpy as np


@dataclasses.dataclass
class Config:
    _target_: str = "imitation_cli.utils.randomness.Config.make"
    seed: int = "${seed}"  # type: ignore

    @staticmethod
    def make(seed: int) -> np.random.Generator:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)

        return np.random.default_rng(seed)