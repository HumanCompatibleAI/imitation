"""Smoke tests for bash scripts in experiments/"""
import os
import subprocess

import pytest


SCRIPT_NAMES = (
    "bc_benchmark.sh"
    "benchmark_and_table.sh"
    "imit_benchmark.sh",
    "train_experts.sh",
    "transfer_learn_benchmark.sh",
)

@pytest.mark.parametrize(
    "script_name",
    SCRIPT_NAMES,
)
def test_experiments_fast(script_name: str):
    """Quickly check that experiments run successfully on fast mode."""
    new_env = dict(os.environ)
    new_env.update(DATA_DIR="tests/data")
    exit_code = subprocess.call([f"./experiments/{script_name}", "--fast"], env=new_env)
    assert exit_code == 0
