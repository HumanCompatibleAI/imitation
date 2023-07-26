import hydra
import pytest

from imitation.algorithms import dagger

# Note: this import is needed to ensure that configurations are properly registered
from imitation_cli.dagger import RunConfig


@pytest.fixture
def dagger_run_config(tmpdir) -> RunConfig:
    """A DAgger run config with a temporary directory as the output directory."""
    with hydra.initialize_config_module(
            version_base=None,
            config_module="imitation_cli.config",
    ):
        yield hydra.compose(
            config_name="dagger_run",
            overrides=[f"hydra.run.dir={tmpdir}"],
            # This is needed to ensure that variable interpolation to hydra.run.dir
            # works properly
            return_hydra_config=True,
        )


@pytest.fixture
def simple_dagger_trainer(dagger_run_config: RunConfig) -> dagger.SimpleDAggerTrainer:
    return hydra.utils.instantiate(dagger_run_config.dagger)


def test_train_dagger_one_step_smoke(simple_dagger_trainer):
    # WHEN
    simple_dagger_trainer.train(1)

    # THEN
    # No exception is raised


