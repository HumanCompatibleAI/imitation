#!/usr/bin/env python
"""Type-check all code-blocks inside Jupyter notebook files.

This relies on nbQA which can be installed without new extra dependencies.
"""
import argparse
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import List

_info = partial(print, file=sys.stderr)


def get_files(input_paths: List) -> List[Path]:
    """Build list of files to scan from list of paths and files."""
    files = []
    for file in input_paths:
        if file.is_dir():
            files.extend(file.glob("**/*.ipynb"))
        else:
            if file.suffix == ".ipynb":
                files.append(file)
            else:
                _info(f"Skipping {file} (not a Jupyter notebook file)")
    if not files:
        _info("No Jupyter notebooks found")
        sys.exit(1)
    return files


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="List of files or paths to check",
    )
    args = parser.parse_args()
    return parser, args


def main():
    """Type-check all code inside Jupyter notebook files."""
    parser, args = parse_args()
    input_paths = get_files(args.files)
    try:
        subprocess.run(["nbqa", "mypy", *input_paths], check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
