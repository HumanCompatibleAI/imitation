"""Tests for config files in imitation/scripts/config/tuned_hps/ folder."""

import pytest

from imitation.scripts import train_adversarial, train_imitation, tuning

ALGORITHMS = ["bc", "dagger", "airl", "gail"]
ENVIRONMENTS = [
    "seals_walker",
    "seals_ant",
    "seals_half_cheetah",
    "seals_hopper",
    "seals_swimmer",
]


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_benchmarks_print_config_succeeds(algorithm: str, environment: str):
    # We test the configs using the print_config command,
    # because running the configs requires MuJoCo.
    # Requiring MuJoCo to run the tests adds too much complexity.

    if algorithm in ("bc", "dagger"):
        experiment = train_imitation.train_imitation_ex
    elif algorithm in ("airl", "gail"):
        experiment = train_adversarial.train_adversarial_ex
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")  # pragma: no cover

    config_name = f"{algorithm}_{environment}"
    run = experiment.run(command_name="print_config", named_configs=[config_name])
    assert run.status == "COMPLETED"


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_tuning_print_config_succeeds(algorithm: str, environment: str):
    # We test the configs using the print_config command,
    # because running the configs requires MuJoCo.
    # Requiring MuJoCo to run the tests adds too much complexity.
    experiment = tuning.tuning_ex
    run = experiment.run(
        command_name="print_config",
        named_configs=[algorithm],
        config_updates=dict(
            parallel_run_config=dict(
                base_named_configs=[f"{algorithm}_{environment}"],
            ),
        ),
    )
    assert run.status == "COMPLETED"
