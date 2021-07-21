import json
import os
import pathlib
import warnings
from typing import Any, Callable, List, NamedTuple, Union

import sacred

from imitation.data import types


class SacredDicts(NamedTuple):
    """Each dict `foo` is loaded from `f"{sacred_dir}/foo.json"`."""

    sacred_dir: str
    config: dict
    run: dict

    @classmethod
    def load_from_dir(cls, sacred_dir: str):
        args = []
        for field in cls._fields:
            if field == "sacred_dir":
                args.append(sacred_dir)
            else:
                json_path = os.path.join(sacred_dir, f"{field}.json")
                with open(json_path, "r") as f:
                    args.append(json.load(f))
        return cls(*args)


def dir_contains_sacred_jsons(dir_path: str) -> bool:
    run_path = os.path.join(dir_path, "run.json")
    config_path = os.path.join(dir_path, "config.json")
    return os.path.isfile(run_path) and os.path.isfile(config_path)


def filter_subdirs(
    root_dir: str,
    filter_fn: Callable[[str], bool] = dir_contains_sacred_jsons,
    *,
    nested_ok: bool = False,
) -> List[str]:
    """Walks through a directory tree, returning paths to filtered subdirectories.

    Does not follow symlinks.

    Args:
      root_dir: The start of the directory tree walk.
      filter_fn: A function with takes a directory path and returns True if
        we should include the directory path in this function's return value.
      nested_ok: If False, then error if in the return value, one of the
        directory paths is a subdirectory of another.

    Returns:
      A list of all subdirectory paths where `filter_fn(path) == True`.
    """
    filtered_dirs = set()
    for root, _, _ in os.walk(root_dir):
        if filter_fn(root):
            filtered_dirs.add(root)

    if not nested_ok:
        for dirpath in filtered_dirs:
            components = os.path.split(dirpath)
            for i in range(1, len(components)):
                prefix = os.path.join(*components[0:i])
                if prefix in filtered_dirs:
                    raise ValueError(f"Parent {prefix} to {dir} also a dir directory")
    return list(filtered_dirs)


def build_sacred_symlink(log_dir: types.AnyPath, run: sacred.run.Run) -> None:
    """Constructs a symlink "{log_dir}/sacred" => "${SACRED_PATH}"."""
    log_dir = pathlib.Path(log_dir)

    sacred_dir = get_sacred_dir_from_run(run)
    if sacred_dir is None:
        warnings.warn(RuntimeWarning("Couldn't find sacred directory."))
        return
    symlink_path = pathlib.Path(log_dir, "sacred")
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
    symlink_path.symlink_to(target_path, target_is_directory=True)


def get_sacred_dir_from_run(run: sacred.run.Run) -> Union[pathlib.Path, None]:
    """Returns path to the sacred directory, or None if not found."""
    for obs in run.observers:
        if isinstance(obs, sacred.observers.FileStorageObserver):
            return pathlib.Path(obs.dir)
    return None


def dict_get_nested(d: dict, nested_key: str, *, sep=".", default=None) -> Any:
    curr = d
    for key in nested_key.split(sep):
        if isinstance(curr, dict) and key in curr:
            curr = curr[key]
        else:
            return default
    return curr
