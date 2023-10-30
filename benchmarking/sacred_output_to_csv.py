"""Converts a directory of Sacred output to a CSV file."""
import pathlib
import sys

from imitation.util.sacred_file_parsing import find_sacred_runs


def main(path: pathlib.Path, only_completed_runs: bool = True):
    if not path.exists():
        raise NotADirectoryError(f"Path {path} does not exist.")

    # Print header
    if only_completed_runs:
        print("algo, env, score, expert_score")
    else:
        print("algo, env, score, expert_score, status")

    # Print data
    for config, run in find_sacred_runs(path, only_completed_runs):
        algo = run["command"]
        env = config["environment"]["gym_id"]
        score = run["result"]["imit_stats"]["monitor_return_mean"]
        expert_score = run["result"]["expert_stats"]["monitor_return_mean"]

        if only_completed_runs:
            print(f"{algo}, {env}, {score}, {expert_score}")
        else:
            status = run["status"]
            print(f"{algo}, {env}, {score}, {expert_score}, {status}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path to sacred output directory>")
        sys.exit(1)

    main(pathlib.Path(sys.argv[1]))
