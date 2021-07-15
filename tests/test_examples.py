import pathlib
import subprocess
from typing import Iterable, List

import pytest
from pytest_notebook import execution, notebook


def _paths_to_strs(x: Iterable[pathlib.Path]) -> List[str]:
    """Convert Path to str for nice Pytest `parameterized` logs.

    For example, if we use Path, we get something inscrutable like
    test_run_example_sh_scripts[sh_path0] rather than seeing the actual path name."""
    return [str(path) for path in x]


THIS_DIR = pathlib.Path(__file__).absolute().parent
EXAMPLES_DIR = THIS_DIR / ".." / "examples"

SH_PATHS = _paths_to_strs(EXAMPLES_DIR.glob("*.sh"))
NB_PATHS = _paths_to_strs(EXAMPLES_DIR.glob("*.ipynb"))
PY_PATHS = _paths_to_strs(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("nb_path", NB_PATHS)
def test_run_example_notebooks(nb_path):
    """Smoke test ensuring that example notebooks run without error.

    The `pytest_notebook` package also includes regression test functionality against
    saved notebook outputs, if we want to check that later.
    """
    nb = notebook.load_notebook(nb_path)
    execution.execute_notebook(nb, cwd=EXAMPLES_DIR, timeout=120)


@pytest.mark.parametrize("py_path", PY_PATHS)
def test_run_example_py_scripts(py_path):
    """Smoke test ensuring that python example scripts run without error."""
    exit_code = subprocess.call(["python", py_path])
    assert exit_code == 0


@pytest.mark.parametrize("sh_path", SH_PATHS)
def test_run_example_sh_scripts(sh_path):
    """Smoke test ensuring that shell example scripts run without error."""
    for _ in range(2):  # Repeat because historically these have failed on second run.
        exit_code = subprocess.call(["env", "bash", "-e", sh_path])
        assert exit_code == 0


README_SNIPPET_PATHS = _paths_to_strs([EXAMPLES_DIR / "quickstart.sh"])


@pytest.mark.parametrize("snippet_path", README_SNIPPET_PATHS)
def test_example_snippets_are_in_readme(snippet_path):
    """Check that README.md examples haven't diverged from snippets."""
    with open(snippet_path, "r") as f:
        x = f.read()
    with open("README.md") as f:
        y = f.read()
    assert x in y, f"{snippet_path} has diverged from README.md"
