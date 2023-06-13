"""Utilities for processing the config files of the benchmarking runs."""

import argparse
import pathlib
from typing import List


def filter_config_files(files: List[str], /) -> List[pathlib.Path]:
    """Filter config files.

    Filter the list of config files to ignore the seed, and only return the last
    file per algorithm and environment.

    Args:
        files: list of config files

    Returns:
        list of config files
    """
    config_files = [pathlib.Path(config_file) for config_file in files]
    config_files = [
        config_file for config_file in config_files if config_file.name == "config.json"
    ]
    # all files have the following format:
    # [/base/path/to/file]/<algo_and_environment>/<other_info>/sacred/1/config.json
    experiments = {}
    for config_file in config_files:
        experiment = config_file.parents[3]
        if experiment not in experiments:
            experiments[experiment] = [config_file]
        else:
            experiments[experiment].append(config_file)
    final_config_files = []
    for experiment, config_files in experiments.items():
        config_files.sort(key=lambda config_file: config_file.parents[1])
        final_config_files.append(config_files[-1])
    return final_config_files


def remove_empty_dicts(d: dict):
    """Remove empty dictionaries in place.

    This is a recursive function that will remove empty dictionaries from a dictionary.

    Args:
        d: dictionary to be filtered
    """
    for key, value in list(d.items()):
        if isinstance(value, dict):
            remove_empty_dicts(value)
            if not value:
                d.pop(key)
        elif value == {}:
            d.pop(key)


def clean_config_file(file: pathlib.Path, write_path: pathlib.Path, /) -> None:
    """Clean a config file.

    reads the file, loads from json to dict, removes keys related to e.g. seeds,
    config paths, leaving only hyperparameters, and writes back to file.

    Args:
        file: path to config file
        write_path: path to write the cleaned config file
    """
    import json

    with open(file) as f:
        config = json.load(f)
    # remove key 'agent_path'
    config.pop("agent_path")
    config.pop("seed")
    config.get("demonstrations", {}).pop("path")
    config.get("expert", {}).get("loader_kwargs", {}).pop("path", None)
    env_name = config.pop("environment").pop("gym_id")
    config["environment"] = {"gym_id": env_name}
    config.pop("show_config", None)

    remove_empty_dicts(config)
    # files are of the format
    # /path/to/file/example_<algo>_<env>_best_hp_eval/<other_info>/sacred/1/config.json
    # we want to write to /<write_path>/<algo>_<env>.json
    with open(write_path / f"{file.parents[3].name}.json", "w") as f:
        json.dump(config, f, indent=4)


def main():
    """Main function of the script."""
    # get two arguments from the terminal. The first positional argument contains the
    # path to a txt file that has a list of paths to config files.
    # The second positional argument is the path to the directory where
    # the cleaned config files should be written.

    parser = argparse.ArgumentParser()
    parser.add_argument("config_files", type=str)
    parser.add_argument("write_path", type=str)
    args = parser.parse_args()

    # read the list of config files from the txt file
    with open(args.config_files) as f:
        config_files_str = f.read().splitlines()
    config_files = [pathlib.Path(file) for file in config_files_str]

    write_path = pathlib.Path(args.write_path)

    # make sure the write path and all the config files already exist
    if not write_path.exists():
        raise ValueError(f"write path {write_path} does not exist")
    for file in config_files:
        if not file.exists():
            raise ValueError(f"config file {file} does not exist")

    for file in config_files:
        clean_config_file(file, write_path)


if __name__ == "__main__":
    main()
