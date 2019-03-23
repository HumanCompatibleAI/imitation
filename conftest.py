"""
Configures pytest to ignore unit tests marked as expensive unless we
use the --expensive flag. The goal of the marking tests as expensive
is to ensure that the default pytest run should take less than 30
seconds.
"""


import pytest


def pytest_addoption(parser):
    parser.addoption("--expensive", action="store_true",
                     help="run expensive tests (which are otherwise skipped).")


def pytest_runtest_setup(item):
    if 'expensive' in item.keywords and not item.config.getoption(
            "--expensive"):
        pytest.skip("Skipping test unless --expensive is flagged")
