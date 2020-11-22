"""Fixtures common across tests."""

import os

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def torch_single_threaded():
    """Make PyTorch and JAX execute code single-threaded.

    This allows us to run the test suite with greater across-test parallelism.
    This is faster, since:
        - There are diminishing returns to more threads within a test.
        - Many tests cannot be multi-threaded (e.g. most not using PyTorch training),
          and we have to set between-test parallelism based on peak resource
          consumption of tests to avoid spurious failures.
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Limit ourselves to single-threaded JAX/XLA operations.
    # See https://github.com/google/jax/issues/743.
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
    )
