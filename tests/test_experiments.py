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


@pytest.mark.parametrize(
    "script_name",
    SCRIPT_NAMES,
)
def test_experiments_fast(script_name: str):
    """Quickly check that experiments run successfully on fast mode."""
    exit_code = subprocess.call([f"./experiments/{script_name}", "--fast"])
    assert exit_code == 0
