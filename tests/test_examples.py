"""Tests examples/*: quickstart code and Jupyter notebook."""

import os
import pathlib
import subprocess
import sys
from typing import Iterable, Sequence

import pytest


def _paths_to_strs(x: Iterable[pathlib.Path]) -> Sequence[str]:
    """Convert Path to str for nice Pytest `parameterized` logs.

    For example, if we use Path, we get something inscrutable like
    test_run_example_sh_scripts[sh_path0] rather than seeing the actual path name.

    Args:
        x: The paths to convert.

    Returns:
        A sequence of the same length as `x`, with each element the string
        representation of the corresponding path in `x`.
    """
    return [str(path) for path in x]


THIS_DIR = pathlib.Path(__file__).absolute().parent
EXAMPLES_DIR = THIS_DIR / ".." / "examples"
TUTORIALS_DIR = THIS_DIR / ".." / "docs" / "tutorials"

SH_PATHS = _paths_to_strs(EXAMPLES_DIR.glob("*.sh"))
PY_PATHS = _paths_to_strs(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("py_path", PY_PATHS)
def test_run_example_py_scripts(py_path):
    """Smoke test ensuring that python example scripts run without error."""
    # We need to use sys.executable, not just "python", on Windows as
    # subprocess.call ignores PATH (unless shell=True) so runs a
    # system-wide Python interpreter outside of our venv. See:
    # https://stackoverflow.com/questions/5658622/
    exit_code = subprocess.call([sys.executable, py_path])
    assert exit_code == 0


@pytest.mark.parametrize("sh_path", SH_PATHS)
def test_run_example_sh_scripts(sh_path):
    """Smoke test ensuring that shell example scripts run without error."""
    if os.name == "nt":  # pragma: no cover
        pytest.skip("bash shell scripts not ported to Windows.")
    for _ in range(2):  # Repeat because historically these have failed on second run.
        exit_code = subprocess.call(["env", "bash", "-e", sh_path])
        assert exit_code == 0


README_SNIPPET_PATHS = _paths_to_strs([EXAMPLES_DIR / "quickstart.sh"])


@pytest.mark.parametrize("snippet_path", README_SNIPPET_PATHS)
def test_example_snippets_are_in_readme(snippet_path):
    """Check that README.md examples haven't diverged from snippets."""
    with open(snippet_path, "r") as f:
        x = "".join(f.readlines()[2:])  # strip away shebang line
    with open("README.md", "r", encoding="utf-8") as f:
        y = f.read()
    assert x in y, f"{snippet_path} has diverged from README.md"
