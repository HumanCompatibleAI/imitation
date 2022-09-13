#!/usr/bin/env python
"""Clean all notebooks in the repository."""
import pathlib

import nbformat


class UncleanNotebookError(Exception):
    """Raised when a notebook is unclean."""


def clean_notebook(file: pathlib.Path, check_only=False) -> None:
    """Clean an ipynb notebook.

    "Cleaning" means removing all output and metadata, as well as any other unnecessary
    or vendor-dependent information or fields, so that it can be committed to the
    repository, and so that artificial diffs are not introduced when the notebook is
    executed.

    Args:
        file: Path to the notebook to clean.
        check_only: If True, only check if the notebook is clean, and raise an
            exception if it is not. If False, clean the notebook in-place.

    Raises:
        UncleanNotebookError: If `check_only` is True and the notebook is not clean.
            Message contains brief description of the reason for the failure.
    """
    # Read the notebook
    with open(file) as f:
        nb = nbformat.read(f, as_version=4)

    # Remove the output and metadata from each cell
    # also reset the execution count
    # if the cell has no code, remove it
    for cell in nb.cells:
        if "outputs" in cell and cell["outputs"]:
            if check_only:
                raise UncleanNotebookError(f"Notebook {file} has outputs")
            cell["outputs"] = []
        if "metadata" in cell and cell["metadata"]:
            if check_only:
                raise UncleanNotebookError(f"Notebook {file} has metadata")
            cell["metadata"] = {}
        if "execution_count" in cell and cell["execution_count"]:
            if check_only:
                raise UncleanNotebookError(f"Notebook {file} has execution count")
            cell["execution_count"] = None
        if cell["cell_type"] == "code" and not cell["source"]:
            if check_only:
                raise UncleanNotebookError(f"Notebook {file} has empty code cell")
            nb.cells.remove(cell)

    if not check_only:
        # Write the notebook
        with open(file, "w") as f:
            nbformat.write(nb, f)


if __name__ == "__main__":
    # if the argument --check has been passed, check if the notebooks are clean
    # otherwise, clean them in-place
    import argparse
    import traceback

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check_only = args.check
    for file in pathlib.Path.cwd().glob("**/*.ipynb"):
        print(f"Cleaning {file}" if not check_only else f"Checking {file}")
        try:
            clean_notebook(file, check_only=check_only)
        except UncleanNotebookError:
            traceback.print_exc()
            exit(1)
