import dataclasses
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING

from imitation.algorithms import dagger
from imitation_cli.utils import environment as environment_cfg, policy_evaluation, randomness, trajectories, policy

from imitation_cli.algorithm_configurations import dagger as dagger_cfg
from imitation_cli.algorithm_configurations import bc as bc_cfg


@dataclasses.dataclass
class RunConfig:
    """Config for running DAgger."""

    rng: randomness.Config = randomness.Config(seed=0)
    total_timesteps: int = int(1e6)
    bc_train_cfg: bc_cfg.TrainConfig = bc_cfg.TrainConfig()
    # checkpoint_interval: int = 0

    environment: environment_cfg.Config = MISSING
    dagger: dagger_cfg.Config = MISSING

    evaluation: policy_evaluation.Config = MISSING
    # This ensures that the working directory is changed
    # to the hydra output dir
    hydra: Any = dataclasses.field(default_factory=lambda: dict(job=dict(chdir=True)))


cs = ConfigStore.instance()

environment_cfg.register_configs("environment", "${rng}")
trajectories.register_configs("dagger/expert_trajs", "${environment}", "${rng}")
policy.register_configs("dagger/bc_trainer/policy", "${environment}")
policy.register_configs("dagger/expert_policy", "${environment}")
policy_evaluation.register_configs("evaluation", "${environment}", "${rng}")

cs.store(
    name="dagger_run_base",
    node=RunConfig(
        dagger=dagger_cfg.Config(
            venv="${environment}",  # type: ignore[arg-type]
            rng="${rng}",  # type: ignore[arg-type]
            bc_trainer=bc_cfg.Config(
                venv="${environment}",  # type: ignore[arg-type]
                rng="${rng}",  # type: ignore[arg-type]

                # Here we ensure DAgger and BC use the same expert trajectories
                demonstrations="${dagger.expert_trajs}",  # type: ignore[arg-type]
            ),
        ),
    ),
)


def run_dagger(cfg: RunConfig):
    dagger_trainer: dagger.DAggerTrainer = instantiate(cfg.dagger)

    dagger_trainer.train(
        total_timesteps=cfg.total_timesteps,
        bc_train_kwargs=cfg.bc_train_cfg,
    )

    dagger_trainer.save_trainer()


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="dagger_run",
)
def main(cfg: RunConfig):
    run_dagger(cfg)


if __name__ == "__main__":
    main()
