#!/usr/bin/env python

"""This script checks that no files in our source code have a "#type: ignore" comment without explicitly indicating the
reason for the ignore. This is to ensure that we don't accidentally ignore errors that we should be fixing."""


import os
import re
import pathlib
import sys
from typing import List

import click

# Regex to match a "# type: ignore" comment not followed by a reason.
TYPE_IGNORE_COMMENT = re.compile(r"#\s*type:\s*ignore\s*(?![^\[]*\[)")

# Regex to match a "# type: ignore[<reason>]" comment.
TYPE_IGNORE_REASON_COMMENT = re.compile(r"#\s*type:\s*ignore\[(?P<reason>.*)\]")


def check_file(file: pathlib.Path):
    """Checks that the given file has no "# type: ignore" comments without a reason."""
    with open(file, "r") as f:
        for i, line in enumerate(f):
            if TYPE_IGNORE_COMMENT.search(line):
                raise ValueError(f"{file}:{i+1}: Found a '# type: ignore' comment without a reason.")

            if search := TYPE_IGNORE_REASON_COMMENT.search(line):
                reason = search.group("reason")
                if reason == "":
                    raise ValueError(f"{file}:{i+1}: Found a '# type: ignore[]' comment without a reason.")


def check_files(files: List[pathlib.Path]):
    """Checks that the given files have no "# type: ignore" comments without a reason."""
    for file in files:
        check_file(file)


def get_files_to_check(root_dir: pathlib.Path) -> List[pathlib.Path]:
    """Returns a list of files that should be checked for "# type: ignore" comments."""
    # Get the list of files that should be checked.
    files = []
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                files.append(pathlib.Path(root) / filename)

    return files


@click.command()
@click.option("--root-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default="src")
def main(root_dir: str):
    """Checks that no files in our source code have a "#type: ignore" comment without explicitly indicating the
    reason for the ignore. This is to ensure that we don't accidentally ignore errors that we should be fixing."""
    files = get_files_to_check(pathlib.Path(root_dir))
    try:
        check_files(files)
    except ValueError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()