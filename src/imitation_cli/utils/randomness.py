import dataclasses


@dataclasses.dataclass
class Config:
    _target_: str = "imitation_cli.utils.randomness.Config.make"
    seed: int = "${seed}"

    @staticmethod
    def make(seed: int):
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)

        return np.random.default_rng(seed)