"""Smoke tests for bash scripts in experiments/"""
import subprocess

import pytest


@pytest.mark.parametrize(
    "script_name",
    ["imit_benchmark.sh", "train_experts.sh", "transfer_learn_benchmark.sh"],
)
def test_experiments_fast(script_name: str):
    """Quickly check that experiments run successfully on fast mode."""
    exit_code = subprocess.call([f"./experiments/{script_name}", "--fast"])
    assert exit_code == 0
