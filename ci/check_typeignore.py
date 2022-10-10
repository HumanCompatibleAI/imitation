#!/usr/bin/env python

"""Check for invalid "# type: ignore" comments.

This script checks that no files in our source code have a "#type: ignore" comment
without explicitly indicating the reason for the ignore. This is to ensure that we
don't accidentally ignore errors that we should be fixing.
"""
import argparse
import os
import pathlib
import re
import sys
from typing import List

# Regex to match a "# type: ignore" comment not followed by a reason.
TYPE_IGNORE_COMMENT = re.compile(r"#\s*type:\s*ignore\s*(?![^\[]*\[)")

# Regex to match a "# type: ignore[<reason>]" comment.
TYPE_IGNORE_REASON_COMMENT = re.compile(r"#\s*type:\s*ignore\[(?P<reason>.*)\]")


class InvalidTypeIgnore(ValueError):
    """Raised when a file has an invalid "# type: ignore" comment."""


def check_file(file: pathlib.Path):
    """Checks that the given file has no "# type: ignore" comments without a reason."""
    with open(file, "r") as f:
        for i, line in enumerate(f):
            if TYPE_IGNORE_COMMENT.search(line):
                raise InvalidTypeIgnore(
                    f"{file}:{i+1}: Found a '# type: ignore' comment without a reason.",
                )

            if search := TYPE_IGNORE_REASON_COMMENT.search(line):
                reason = search.group("reason")
                if reason == "":
                    raise InvalidTypeIgnore(
                        f"{file}:{i+1}: Found a '# type: ignore[]' "
                        "comment without a reason.",
                    )


def check_files(files: List[pathlib.Path]):
    """Checks that the given files have no type: ignore comments without a reason."""
    for file in files:
        if file == pathlib.Path(__file__):
            continue
        check_file(file)


def get_files_to_check(root_dirs: List[pathlib.Path]) -> List[pathlib.Path]:
    """Returns a list of files that should be checked for "# type: ignore" comments."""
    # Get the list of files that should be checked.
    files = []
    for root_dir in root_dirs:
        for root, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".py"):
                    files.append(pathlib.Path(root) / filename)

    return files


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="+",
        type=pathlib.Path,
        help="List of files or paths to check for invalid '# type: ignore' comments.",
    )
    args = parser.parse_args()
    return parser, args


def main():
    """Check for invalid "# type: ignore" comments."""
    parser, args = parse_args()
    file_list = get_files_to_check(args.files)
    try:
        check_files(file_list)
    except InvalidTypeIgnore as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
