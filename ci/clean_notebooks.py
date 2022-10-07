#!/usr/bin/env python
"""Clean all notebooks in the repository."""
import argparse
import pathlib
import traceback
from typing import Any, List, Tuple

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

    was_dirty = False

    if check_only:
        print(f"Checking {file}")

    # Remove the output and metadata from each cell
    # also reset the execution count
    # if the cell has no code, remove it
    fields_defaults: List[Tuple[str, Any]] = [
        ("execution_count", None),
        ("outputs", []),
        ("metadata", {}),
    ]
    for cell in nb.cells:
        if cell["cell_type"] == "code" and not cell["source"]:
            if check_only:
                raise UncleanNotebookError(f"Notebook {file} has empty code cell")
            nb.cells.remove(cell)
            was_dirty = True
        for field, default in fields_defaults:
            if cell.get(field) != default:
                was_dirty = True
                if check_only:
                    raise UncleanNotebookError(
                        f"Notebook {file} is not clean: cell has "
                        f"field {field!r} with value {cell[field]!r} (expected "
                        f"{default!r}). Cell:\n{cell['source']!r}",
                    )
                else:
                    cell[field] = default

    if not check_only and was_dirty:
        # Write the notebook
        with open(file, "w") as f:
            nbformat.write(nb, f)
        print(f"Cleaned {file}")

def main():
    """Clean all notebooks in the repository, or check that they are clean."""
    # if the argument --check has been passed, check if the notebooks are clean
    # otherwise, clean them in-place
    parser = argparse.ArgumentParser()
    # capture files and paths to clean
    parser.add_argument(
        "files",
        nargs="+",
        type=pathlib.Path,
        help="List of files or paths to clean",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check_only = args.check
    # build list of files to scan from list of paths and files
    files = []
    if len(args.files) == 0:
        parser.print_help()
        exit(1)
    for file in args.files:
        if file.is_dir():
            files.extend(file.glob("**/*.ipynb"))
        else:
            if file.suffix == ".ipynb":
                files.append(file)
            else:
                print(f"Skipping {file} (not a notebook)")
    if not files:
        print("No notebooks found")
        exit(1)
    for file in files:
        try:
            clean_notebook(file, check_only=check_only)
        except UncleanNotebookError:
            traceback.print_exc()
            exit(1)


if __name__ == "__main__":
    main()
