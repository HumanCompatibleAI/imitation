#!/usr/bin/env python
"""Clean all notebooks in the repository."""
import argparse
import pathlib
import sys
import traceback
from typing import Any, Dict, List

import nbformat


class UncleanNotebookError(Exception):
    """Raised when a notebook is unclean."""


markdown_structure: Dict[str, Dict[str, Any]] = {
    "cell_type": {"do": "keep"},
    "metadata": {"do": "constant", "value": dict()},
    "source": {"do": "keep"},
    "id": {"do": "keep"},
    "attachments": {"do": "constant", "value": {}},
}

code_structure: Dict[str, Dict[str, Any]] = {
    "cell_type": {"do": "keep"},
    "metadata": {"do": "constant", "value": dict()},
    "source": {"do": "keep"},
    "outputs": {"do": "constant", "value": list()},
    "execution_count": {"do": "constant", "value": None},
    "id": {"do": "keep"},
}

structure: Dict[str, Dict[str, Dict[str, Any]]] = {
    "markdown": markdown_structure,
    "code": code_structure,
}


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
        ValueError: unknown cell structure action.
    """
    # Read the notebook
    with open(file) as f:
        nb = nbformat.read(f, as_version=4)

    was_dirty = False

    if check_only:
        print(f"Checking {file}")

    for cell in nb.cells:
        # Remove empty cells
        if cell["cell_type"] == "code" and not cell["source"]:
            if check_only:
                raise UncleanNotebookError(f"Notebook {file} has empty code cell")
            nb.cells.remove(cell)
            was_dirty = True

        # Clean the cell
        # (copy the cell keys list so we can iterate over it while modifying it)
        for key in list(cell):
            if key not in structure[cell["cell_type"]]:
                if check_only:
                    raise UncleanNotebookError(
                        f"Notebook {file} has unknown cell key {key} for cell type "
                        + f"{cell['cell_type']}",
                    )
                del cell[key]
                was_dirty = True
            else:
                cell_structure = structure[cell["cell_type"]][key]
                if cell_structure["do"] == "keep":
                    continue
                elif cell_structure["do"] == "constant":
                    constant_value = cell_structure["value"]
                    if cell[key] != constant_value:
                        if check_only:
                            raise UncleanNotebookError(
                                f"Notebook {file} has illegal cell value for key {key}"
                                f" (value: {cell[key]}, "
                                f"expected: {constant_value})",
                            )
                        cell[key] = constant_value
                        was_dirty = True
                else:
                    raise ValueError(
                        f"Unknown cell structure action {cell_structure['do']}",
                    )

    if not check_only and was_dirty:
        # Write the notebook
        with open(file, "w") as f:
            nbformat.write(nb, f)
        print(f"Cleaned {file}")


def parse_args():
    """Parse command-line arguments.

    Returns:
        parser: The parser object.
        args: The parsed arguments.
    """
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
    return parser, args


def get_files(input_paths: List):
    """Build list of files to scan from list of paths and files.

    Args:
        input_paths: List of paths and files to scan.

    Returns:
        files: List of files to scan.
    """
    files = []
    for file in input_paths:
        if file.is_dir():
            files.extend(file.glob("**/*.ipynb"))
        else:
            if file.suffix == ".ipynb":
                files.append(file)
            else:
                print(f"Skipping {file} (not a notebook)")
    if not files:
        print("No notebooks found")
        sys.exit(1)
    return files


def main():
    """Clean all notebooks in the repository, or check that they are clean."""
    parser, args = parse_args()
    check_only = args.check
    input_paths = args.files

    if len(input_paths) == 0:
        parser.print_help()
        sys.exit(1)

    files = get_files(input_paths)

    for file in files:
        try:
            clean_notebook(file, check_only=check_only)
        except UncleanNotebookError:
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
