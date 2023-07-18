"""Tests for config files in benchmarking/ folder."""
import pathlib
import subprocess
import sys

import pytest

from imitation.scripts import train_adversarial, train_imitation

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
        BENCHMARKING_DIR / f"{algorithm}_{environment}_best_hp_eval.json",
    )

    # WHEN
    experiment.add_named_config(config_name, config_file)
    run = experiment.run(command_name="print_config", named_configs=[config_name])

    # THEN
    assert run.status == "COMPLETED"


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_tuning_print_config_succeeds(algorithm: str):
    # We test the configs using the print_config command,
    # because running the configs requires MuJoCo.
    # Requiring MuJoCo to run the tests adds too much complexity.

    # We need to use sys.executable, not just "python", on Windows as
    # subprocess.call ignores PATH (unless shell=True) so runs a
    # system-wide Python interpreter outside of our venv. See:
    # https://stackoverflow.com/questions/5658622/
    tuning_path = str(BENCHMARKING_DIR / "tuning.py")
    env = 'parallel_run_config.base_named_configs=["seals_cartpole"]'
    exit_code = subprocess.call(
        [
            sys.executable,
            tuning_path,
            "print_config",
            "with",
            f"{algorithm}",
            env,
        ],
    )
    assert exit_code == 0
