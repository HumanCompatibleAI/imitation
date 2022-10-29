"""Smoke tests for bash scripts in experiments/."""

import os
import subprocess

import pytest

SCRIPT_NAMES = (
    "bc_benchmark.sh",
    "dagger_benchmark.sh",
    "benchmark_and_table.sh",
    "imit_benchmark.sh",
    "rollouts_from_policies.sh",
    "transfer_learn_benchmark.sh",
)


@pytest.mark.parametrize(
    "script_name",
    SCRIPT_NAMES,
)
def test_experiments_fast(script_name: str):
    """Quickly check that experiments run successfully on fast mode."""
    if os.name == "nt":  # pragma: no cover
        pytest.skip("bash shell scripts not ported to Windows.")
    exit_code = subprocess.call([f"./experiments/{script_name}", "--fast"])
    assert exit_code == 0
