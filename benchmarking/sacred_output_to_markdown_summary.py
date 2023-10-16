"""Generate a markdown summary of the results of a benchmarking run."""
import pathlib
import sys
from collections import Counter

from imitation.util.sacred_file_parsing import (
    find_sacred_runs,
    group_runs_by_algo_and_env,
)


def print_markdown_summary(path: pathlib.Path):
    if not path.exists():
        raise NotADirectoryError(f"Path {path} does not exist.")

    print("# Benchmark Summary")
    runs_by_algo_and_env = group_runs_by_algo_and_env(path)
    algos = sorted(runs_by_algo_and_env.keys())

    print("## Run status" "")
    print("Status | Count")
    print("--- | ---")
    status_counts = Counter((run["status"] for _, run in find_sacred_runs(path)))
    statuses = sorted(list(status_counts))
    for status in statuses:
        print(f"{status} | {status_counts[status]}")
    print()

    print("## Detailed Run Status")
    print(f"Algorithm | Environment | {' | '.join(sorted(list(status_counts)))}")
    print("--- | --- " + " | --- " * len(statuses))
    for algo in algos:
        envs = sorted(runs_by_algo_and_env[algo].keys())
        for env in envs:
            status_counts = Counter(
                (run["status"] for run in runs_by_algo_and_env[algo][env]),
            )
            print(
                f"{algo} | {env} | "
                f"{' | '.join([str(status_counts[status]) for status in statuses])}",
            )
    print()
    print("## Raw Scores")
    print()
    for algo in algos:
        print(f"### {algo.upper()}")
        print("Environment | Scores | Expert Scores")
        print("--- | --- | ---")
        envs = sorted(runs_by_algo_and_env[algo].keys())
        for env in envs:
            completed_runs = [
                run
                for run in runs_by_algo_and_env[algo][env]
                if run["status"] == "COMPLETED"
            ]
            algo_scores = [
                run["result"]["imit_stats"]["monitor_return_mean"]
                for run in completed_runs
            ]
            expert_scores = [
                run["result"]["expert_stats"]["monitor_return_mean"]
                for run in completed_runs
            ]
            print(
                f"{env} | "
                f"{', '.join([f'{score:.2f}' for score in algo_scores])} | "
                f"{', '.join([f'{score:.2f}' for score in expert_scores])}",
            )
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path to sacred run folder>")
        sys.exit(1)

    print_markdown_summary(pathlib.Path(sys.argv[1]))
