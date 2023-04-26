"""Config and run configuration for AIRL."""
import dataclasses
import logging
import pathlib
from typing import Any, Dict, Sequence, cast

import hydra
import torch as th
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING

from imitation.policies import serialize
from imitation_cli.algorithm_configurations import airl as airl_cfg
from imitation_cli.utils import environment as environment_cfg
from imitation_cli.utils import (
    policy_evaluation,
    randomness,
    reward_network,
    rl_algorithm,
    trajectories,
)


@dataclasses.dataclass
class RunConfig:
    """Config for running AIRL."""

    rng: randomness.Config = randomness.Config(seed=0)
    total_timesteps: int = int(1e6)
    checkpoint_interval: int = 0

    environment: environment_cfg.Config = MISSING
    airl: airl_cfg.Config = MISSING
    evaluation: policy_evaluation.Config = MISSING
    # This ensures that the working directory is changed
    # to the hydra output dir
    hydra: Any = dataclasses.field(default_factory=lambda: dict(job=dict(chdir=True)))


cs = ConfigStore.instance()
environment_cfg.register_configs("environment", "${rng}")
trajectories.register_configs("airl/demonstrations", "${environment}", "${rng}")
rl_algorithm.register_configs("airl/gen_algo", "${environment}", "${rng.seed}")
reward_network.register_configs("airl/reward_net", "${environment}")
policy_evaluation.register_configs("evaluation", "${environment}", "${rng}")

cs.store(
    name="airl_run_base",
    node=RunConfig(
        airl=airl_cfg.Config(
            venv="${environment}",  # type: ignore
        ),
    ),
)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="airl_run",
)
def run_airl(cfg: RunConfig) -> Dict[str, Any]:
    from imitation.algorithms.adversarial import airl
    from imitation.data import rollout
    from imitation.data.types import TrajectoryWithRew

    trainer: airl.AIRL = instantiate(cfg.airl)

    checkpoints_path = pathlib.Path("checkpoints")

    def save(path: str):
        """Save discriminator and generator."""
        # We implement this here and not in Trainer since we do not want to actually
        # serialize the whole Trainer (including e.g. expert demonstrations).
        save_path = checkpoints_path / path
        save_path.mkdir(parents=True, exist_ok=True)

        th.save(trainer.reward_train, save_path / "reward_train.pt")
        th.save(trainer.reward_test, save_path / "reward_test.pt")
        serialize.save_stable_model(save_path / "gen_policy", trainer.gen_algo)

    def callback(round_num: int, /) -> None:
        if cfg.checkpoint_interval > 0 and round_num % cfg.checkpoint_interval == 0:
            logging.log(logging.INFO, f"Saving checkpoint at round {round_num}")
            save(f"{round_num:05d}")

    trainer.train(cfg.total_timesteps, callback)
    imit_stats = policy_evaluation.eval_policy(trainer.policy, cfg.evaluation)

    # Save final artifacts.
    if cfg.checkpoint_interval >= 0:
        logging.log(logging.INFO, "Saving final checkpoint.")
        save("final")

    return {
        "imit_stats": imit_stats,
        "expert_stats": rollout.rollout_stats(
            cast(Sequence[TrajectoryWithRew], trainer.get_demonstrations()),
        ),
    }


if __name__ == "__main__":
    run_airl()
