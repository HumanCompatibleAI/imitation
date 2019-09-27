# Copyright 2019 Google LLC.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Callable, List, NamedTuple
import warnings

import sacred


class SacredDicts(NamedTuple):
  """Each dict `foo` is loaded from `f"{sacred_dir}/foo.json"`."""
  sacred_dir: str
  config: dict
  result: dict
  run: dict

  @classmethod
  def load_from_dir(cls, sacred_dir: str):
    args = []
    for field in SacredDicts._fields:
      if field == "sacred_dir":
        args.append(dir)
      else:
        json_path = os.path.join(dir, "sacred", f"{field}.json")
        if field == "result" and not os.path.is_file(json_path):
          args.append({})  # "result.json" isn't written if it's empty.
        else:
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
  """Walks through a directory tree, return paths to filtered directories.

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


def build_sacred_symlink(log_dir: str, run: sacred.run.Run) -> None:
  """Constructs a symlink "{log_dir}/sacred" => "${SACRED_PATH}"."""
  sacred_dir = get_sacred_dir_from_run(run)
  if sacred_dir is None:
    warnings.warn(RuntimeWarning("Couldn't find sacred directory."))
    return
  symlink_path = os.path.join(log_dir, "sacred")
  os.symlink(sacred_dir, symlink_path)


def get_sacred_dir_from_run(run: sacred.run.Run) -> str:
  for obs in run.observers:
    if isinstance(obs, sacred.FileStorageObserver):
      return obs.dir
  return None
