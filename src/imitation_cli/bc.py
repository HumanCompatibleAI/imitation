import dataclasses
from typing import Any, Optional, cast, Sequence

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew
from imitation_cli.utils import environment as environment_cfg, policy_evaluation, randomness, trajectories, policy

from imitation_cli.algorithm_configurations import bc as bc_cfg


@dataclasses.dataclass
class RunConfig:
    """Config for running BC."""

    rng: randomness.Config = randomness.Config(seed=0)
    environment: environment_cfg.Config = MISSING

    bc: bc_cfg.Config = bc_cfg.Config()

    bc_train_config: bc_cfg.TrainConfig = bc_cfg.TrainConfig()
    log_interval: int = 10  #TODO: find proper default

    evaluation: policy_evaluation.Config = MISSING
    # This ensures that the working directory is changed
    # to the hydra output dir
    hydra: Any = dataclasses.field(default_factory=lambda: dict(job=dict(chdir=True)))


cs = ConfigStore.instance()

environment_cfg.register_configs("environment", "${rng}")
trajectories.register_configs("bc/demonstrations", "${environment}", "${rng}")
policy.register_configs("bc/policy", "${environment}")
policy_evaluation.register_configs("evaluation", "${environment}", "${rng}")

cs.store(
    name="bc_run_base",
    node=RunConfig(
        bc=bc_cfg.Config(
            venv="${environment}",  # type: ignore[arg-type]
            rng="${rng}",  # type: ignore[arg-type]
        ),
    ),
)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="bc_run",
)
def run_bc(cfg: RunConfig):
    trainer: bc.BC = instantiate(cfg.bc)

    trainer.train(
        **cfg.bc_train_config,
        log_interval=cfg.log_interval,
    )

    imit_stats = policy_evaluation.eval_policy(trainer.policy, cfg.evaluation)

    return {
        "imit_stats": imit_stats,
        "expert_stats": rollout.rollout_stats(
            cast(Sequence[TrajectoryWithRew], trainer.get_demonstrations()),
        ),
    }


if __name__ == "__main__":
    run_bc()
