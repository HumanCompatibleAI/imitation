"""Tests for config files in benchmarking/ folder."""
import pathlib

import pytest

from imitation.scripts import train_adversarial
from imitation.scripts.config import train_imitation

THIS_DIR = pathlib.Path(__file__).absolute().parent
BENCHMARKING_DIR = THIS_DIR.parent / "benchmarking"

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

    # GIVEN
    if algorithm in ("bc", "dagger"):
        experiment = train_imitation.train_imitation_ex
    elif algorithm in ("airl", "gail"):
        experiment = train_adversarial.train_adversarial_ex
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")  # pragma: no cover

    config_name = f"{algorithm}_{environment}"
    config_file = str(
        BENCHMARKING_DIR / f"example_{algorithm}_{environment}_best_hp_eval.json",
    )

    # WHEN
    experiment.add_named_config(config_name, config_file)
    run = experiment.run(command_name="print_config", named_configs=[config_name])

    # THEN
    assert run.status == "COMPLETED"
