"""Helper methods for the `sacred` experimental configuration and logging framework."""

import json
import os
import pathlib
import warnings
from typing import Any, Callable, NamedTuple, Optional, Sequence

import sacred
import sacred.observers
import sacred.run

from imitation.data import types
from imitation.util import util


class SacredDicts(NamedTuple):
    """Each dict `foo` is loaded from `f"{sacred_dir}/foo.json"`."""

    sacred_dir: pathlib.Path
    config: dict
    run: dict

    @classmethod
    def load_from_dir(cls, sacred_dir: pathlib.Path):
        return cls(
            sacred_dir=sacred_dir,
            config=json.loads((sacred_dir / "config.json").read_text()),
            run=json.loads((sacred_dir / "run.json").read_text()),
        )


def dir_contains_sacred_jsons(dir_path: pathlib.Path) -> bool:
    run_path = dir_path / "run.json"
    config_path = dir_path / "config.json"
    return run_path.is_file() and config_path.is_file()


def filter_subdirs(
    root_dir: pathlib.Path,
    filter_fn: Callable[[pathlib.Path], bool] = dir_contains_sacred_jsons,
    *,
    nested_ok: bool = False,
) -> Sequence[pathlib.Path]:
    """Walks through a directory tree, returning paths to filtered subdirectories.

    Does not follow symlinks.

    Args:
        root_dir: The start of the directory tree walk.
        filter_fn: A function with takes a directory path and returns True if
            we should include the directory path in this function's return value.
        nested_ok: Allow returning "nested" directories, i.e. a return value where
            some elements are subdirectories of other elements.

    Returns:
        A list of all subdirectory paths where `filter_fn(path) == True`.

    Raises:
        ValueError: If `nested_ok` is False and one of the filtered directory
            paths is a subdirecotry of another.
    """
    filtered_dirs = set()
    for root_str, _, _ in os.walk(root_dir, followlinks=False):
        root = pathlib.Path(root_str)
        if filter_fn(root):
            filtered_dirs.add(root)

    if not nested_ok:
        for dirpath in filtered_dirs:
            for other_dirpath in filtered_dirs:
                if dirpath != other_dirpath and other_dirpath in dirpath.parents:
                    raise ValueError(
                        f"Found nested directories: {dirpath} and {other_dirpath}",
                    )
    return list(filtered_dirs)


def build_sacred_symlink(log_dir: types.AnyPath, run: sacred.run.Run) -> None:
    """Constructs a symlink "{log_dir}/sacred" => "${SACRED_PATH}"."""
    log_dir = util.parse_path(log_dir)

    sacred_dir = get_sacred_dir_from_run(run)
    if sacred_dir is None:
        warnings.warn(RuntimeWarning("Couldn't find sacred directory."))
        return
    symlink_path = log_dir / "sacred"
    target_path = pathlib.Path(os.path.relpath(sacred_dir, start=log_dir))

    # Path.symlink_to errors if the symlink already exists. In our case, we actually
    # want to override the symlink to point to the most recent Sacred dir. The
    # examples/quickstart.sh script fails without this check when run a second time.
    #
    # If `symlink_path` exists and is not a symlink, then it was created by something
    # other than this function then we don't remove it (and will error on the symlink
    # step).
    if symlink_path.is_symlink():
        symlink_path.unlink()

    # Use relative paths so we can mount the output directory at different paths
    # (e.g. when copying across machines).
    try:
        symlink_path.symlink_to(target_path, target_is_directory=True)
    except OSError as e:
        if os.name == "nt":  # Windows
            msg = (
                "Exception occurred while creating symlink. "
                "Please ensure that Developer mode is enabled."
            )
            raise OSError(msg) from e
        else:
            raise e


def get_sacred_dir_from_run(run: sacred.run.Run) -> Optional[pathlib.Path]:
    """Returns path to the sacred directory, or None if not found."""
    for obs in run.observers:
        if isinstance(obs, sacred.observers.FileStorageObserver):
            return util.parse_path(obs.dir)
    return None


def dict_get_nested(d: dict, nested_key: str, *, sep=".", default=None) -> Any:
    curr = d
    for key in nested_key.split(sep):
        if isinstance(curr, dict) and key in curr:
            curr = curr[key]
        else:
            return default
    return curr
