import hydra
import pytest

from imitation.algorithms.adversarial import airl

# Note: this import is needed to ensure that configurations are properly registered
from imitation_cli.airl import RunConfig


@pytest.fixture
def airl_run_config(tmpdir) -> RunConfig:
    """A AIRL run config with a temporary directory as the output directory."""
    with hydra.initialize_config_module(
            version_base=None,
            config_module="imitation_cli.config",
    ):
        yield hydra.compose(
            config_name="airl_run",
            overrides=[f"hydra.run.dir={tmpdir}"],
            # This is needed to ensure that variable interpolation to hydra.run.dir
            # works properly
            return_hydra_config=True,
        )


@pytest.fixture
def airl_trainer(airl_run_config: RunConfig) -> airl.AIRL:
    return hydra.utils.instantiate(airl_run_config.airl)


def test_train_airl_trainer_some_steps_smoke(airl_trainer: airl.AIRL):
    # WHEN
    # Note: any value lower than 16386 will raise an exception
    airl_trainer.train(16386)

    # THEN
    # No exception is raised


