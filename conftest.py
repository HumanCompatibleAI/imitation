import pytest

def pytest_addoption(parser):
        parser.addoption("--expensive", action="store_true",
        help="run expensive tests (which are otherwise skipped).")

def pytest_runtest_setup(item):
    if 'expensive' in item.keywords and not item.config.getoption(
            "--expensive"):
        pytest.skip("Skipping test unless --expensive is flagged")
