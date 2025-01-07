#!/usr/bin/env python
"""Type-check all code-blocks inside rst documentation files."""
import argparse
import sys
import tempfile
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Dict, List

from docutils.core import publish_doctree  # type: ignore
from docutils.nodes import literal_block
from mypy import api

_TMP_DIR = Path(tempfile.gettempdir()) / "imitation" / "typecheck"
_REPO_DIR = Path(__file__).parent.parent


_info = partial(print, file=sys.stderr)


def get_files(input_paths: List) -> List[Path]:
    """Build list of files to scan from list of paths and files."""
    files = []
    for file in input_paths:
        if file.is_dir():
            files.extend(file.glob("**/*.rst"))
        else:
            if file.suffix == ".rst":
                files.append(file)
            else:
                _info(f"Skipping {file} (not a documentation file)")
    if not files:
        _info("No documentation files found")
        sys.exit(1)
    return files


def get_code_blocks(file: Path) -> Dict[int, str]:
    """Find all Python code-blocks inside an rst documentation file.

    Args:
        file: The rst documentation file to scan.

    Returns:
        Mapping from line number to Python code block.
    """
    rst_content = file.read_text()
    doc_parse_f = StringIO()
    document = publish_doctree(
        rst_content,
        settings_overrides={"warning_stream": doc_parse_f},
    )

    python_blocks = {}
    for node in document.traverse(literal_block):
        if "code" in node.get("classes") and "python" in node.get("classes"):
            src_text = node.astext()
            end_line = node.line  # node.line = line number of the end of the block
            start_line = end_line - len(src_text.split("\n"))
            python_blocks[start_line] = src_text

    return python_blocks


def typecheck_doc_file(file: Path) -> List[str]:
    """Type-check Python code-blocks inside an rst documentation file using pytype/mypy.

    Args:
        file: The rst documentation file to type-check.

    Returns:
        List of type errors (str) in the documentation code-blocks.
    """
    code_blocks = get_code_blocks(file)
    file = file.relative_to(_REPO_DIR)
    tmp = _TMP_DIR / file
    tmp.parent.mkdir(parents=True, exist_ok=True)

    errors = []
    for line, code_block in code_blocks.items():
        temp_file = tmp.with_suffix(f".{line}.py")
        temp_file.write_text(code_block)
        file_errors = mypy_codeblock(temp_file)

        def post_process_error_msg(error_msg: str) -> str:
            """Change the error message to use the original file path and line number.

            Replaces temp_file path with original path in error_msg and
            recalculates the line number.

            Args:
                error_msg: The error message to post-process.

            Returns:
                The post-processed error message in the standard mypy format.
            """
            try:
                path, line_no, *rest = error_msg.split(":")
                return ":".join([str(file), str(line + int(line_no) - 1), *rest])
            except ValueError:
                # error_msg is not a std mypy error message
                return error_msg

        errors += [post_process_error_msg(msg) for msg in file_errors]
    return errors


def mypy_codeblock(codeblock: Path) -> List:
    stdout, stderr, exit_status = api.run([str(codeblock)])
    if exit == 0 or not stdout or "no issues found" in stdout:
        return []
    # format of stdout output:
    # /<path>:6: error: Name "policy" is not defined  [name-defined]
    # /<path>:8: error: Too many positional arguments for "register" ...
    # Found 2 errors in 1 file (checked 1 source file)
    return stdout.strip().split("\n")[:-1]  # last line is redundant


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
    """Type-check all code-blocks inside rst documentation files."""
    parser, args = parse_args()
    input_paths = args.files

    if len(input_paths) == 0:
        parser.print_help()
        sys.exit(1)

    files = get_files(input_paths)

    errors = []
    affected_files = 0
    for file in files:
        if file_errors := typecheck_doc_file(file):
            errors += file_errors
            affected_files += 1
            _info(f"{file}: {len(file_errors)} error{'s'[:len(file_errors)^1]}")
        else:
            _info(f"{file}: OK")

    f = len(files)
    e = len(errors)
    a = affected_files
    print("\n".join(errors))
    _info(
        f"Found {e} error{'s'[:e^1]} in {a} file{'s'[:a^1]}"
        f" (checked {f} source file{'s'[:f^1]}).",
    )
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
