"""Smoke tests for bash scripts in experiments/."""

import os
import pathlib
import subprocess

import pytest

SCRIPT_NAMES = (
    "bc_benchmark.sh",
    "dagger_benchmark.sh",
    "benchmark_and_table.sh",
    "imit_benchmark.sh",
    "rollouts_from_policies.sh",
    "train_experts.sh",
    "transfer_learn_benchmark.sh",
)

USES_FULL_ROLLOUTS = ("benchmark_and_table.sh",)

_test_path = pathlib.Path(
    "data",
    "expert_models",
    "half_cheetah_0",
    "rollouts",
    "final.pkl",
)

HAS_FULL_ROLLOUTS = _test_path.exists()


@pytest.mark.parametrize(
    "script_name",
    SCRIPT_NAMES,
)
def test_experiments_fast(script_name: str):
    """Quickly check that experiments run successfully on fast mode."""
    if os.name == "nt":  # pragma: no cover
        pytest.skip("bash shell scripts not ported to Windows.")

    env = None
    if script_name in USES_FULL_ROLLOUTS:
        if not HAS_FULL_ROLLOUTS:
            pytest.skip("Need to download or generate benchmark demonstrations first.")
    else:
        test_data_env = dict(os.environ)
        test_data_env.update(DATA_DIR="tests/testdata")
        env = test_data_env

    exit_code = subprocess.call([f"./experiments/{script_name}", "--fast"], env=env)
    assert exit_code == 0
