"""Utilities to parse sacred run directories."""
import json
import pathlib
import warnings
from collections import defaultdict
from typing import Any, Dict, Generator, List, Tuple

SacredRun = Dict[str, Any]
SacredConfAndRun = Tuple[Dict[str, Any], SacredRun]
GroupedRuns = Dict[str, Dict[str, List[SacredRun]]]


def find_sacred_runs(
    run_path: pathlib.Path,
    only_completed_runs: bool = False,
) -> Generator[SacredConfAndRun, None, None]:
    """Recursively iterates the sacred runs found below the given path.

    Assumes runs in the format of the sacred FileStorageObserver: each run consists
    of a folder that contains a config.json and a run.json file.

    Note: will work with nested directories and can therefore be applied to the
    `output/sacred` folder of the command line interface which creates sub-folders for
    each script.

    Args:
        run_path: The path to search for sacred run directories.
        only_completed_runs: If True, only yields runs that have a run.json file with
            status "COMPLETED".

    Yields:
        Tuples of (config, run) dicts.
    """
    for config_path in run_path.rglob("config.json"):
        run_path = config_path.parent / "run.json"

        if run_path.exists():
            run = json.loads(run_path.read_text())
            if only_completed_runs and run["status"] != "COMPLETED":
                continue
            conf = json.loads(config_path.read_text())
            yield conf, run
        else:
            warnings.warn(f"Run {config_path.parent} has no run.json")


def group_runs_by_algo_and_env(
    path: pathlib.Path,
    only_completed_runs: bool = False,
) -> GroupedRuns:
    """Groups the runs found below the given path by algorithm and environment.

    Access all the runs of algorithm `algo` and environment `env` via
    `runs_by_algo_and_env[algo][env]`.

    Args:
        path: The path to search for sacred run directories.
        only_completed_runs: If True, only yields runs that have a run.json file with
            status "COMPLETED".

    Returns:
        A dictionary mapping algorithms to environments to lists of runs.
    """
    runs_by_algo_and_env: GroupedRuns = defaultdict(lambda: defaultdict(list))
    for conf, run in find_sacred_runs(path, only_completed_runs):
        algo = run["command"]
        env = conf["environment"]["gym_id"]
        runs_by_algo_and_env[algo][env].append(run)

    return runs_by_algo_and_env
