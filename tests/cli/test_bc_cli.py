import hydra
import pytest

from imitation.algorithms import bc

# Note: this import is needed to ensure that configurations are properly registered
from imitation_cli.bc import RunConfig


@pytest.fixture
def bc_run_config(tmpdir) -> RunConfig:
    """A BC run config with a temporary directory as the output directory."""
    with hydra.initialize_config_module(
            version_base=None,
            config_module="imitation_cli.config",
    ):
        yield hydra.compose(
            config_name="bc_run",
            overrides=[f"hydra.run.dir={tmpdir}"],
            # This is needed to ensure that variable interpolation to hydra.run.dir
            # works properly
            return_hydra_config=True,
        )


@pytest.fixture
def bc_trainer(bc_run_config: RunConfig) -> bc.BC:
    return hydra.utils.instantiate(bc_run_config.bc)


def test_train_bc_trainer_one_batch_smoke(bc_trainer: bc.BC):
    # WHEN
    bc_trainer.train(n_batches=1)

    # THEN
    # No exception is raised


