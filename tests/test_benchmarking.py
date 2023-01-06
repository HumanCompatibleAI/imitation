"""Tests for config files in benchmarking/ folder."""
import os
import subprocess

import pytest

from imitation.data import types

ALGO_FAST_CONFIGS = {
    "adversarial": [
        "environment.fast",
        "demonstrations.fast",
        "rl.fast",
        "train.fast",
        "fast",
    ],
    "eval_policy": ["environment.fast", "fast"],
    "imitation": ["environment.fast", "demonstrations.fast", "train.fast", "fast"],
    "preference_comparison": ["environment.fast", "rl.fast", "train.fast", "fast"],
    "rl": ["environment.fast", "rl.fast", "train.fast", "fast"],
}

BENCHMARKING_DIR = types.parse_path("benchmarking")

if not BENCHMARKING_DIR.exists():  # pragma: no cover
    raise RuntimeError(
        "The benchmarking/ folder has not been found. Make sure you are "
        "running tests relative to the base imitation project folder.",
    )


@pytest.mark.parametrize(
    "script_name, command_name",
    [
        ("train_imitation", "bc"),
        ("train_imitation", "dagger"),
        ("train_adversarial", "airl"),
        ("train_adversarial", "gail"),
    ],
)
def test_benchmarking_configs(tmpdir, script_name, command_name):
    # We test that the configs using the print_config command
    # only for the half_cheetah environment,
    # because the print_config command is slow
    # (takes about 5 seconds per execution).
    # We do not test that the configs run even with the fast configs applied,
    # because running the configs requires MuJoCo. Requiring MuJoCo to run
    # the tests adds too much complexity.
    config_file = f"example_{command_name}_seals_half_cheetah_best_hp_eval.json"
    config_path = os.path.join(BENCHMARKING_DIR.stem, config_file)
    completed_process = subprocess.run(
        f"python -m imitation.scripts.{script_name} print_config with {config_path}",
        shell=True,
        capture_output=False,
        stdin=subprocess.DEVNULL,
        check=True,
    )
    assert completed_process.returncode == 0
