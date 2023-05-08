"""Compare experiment results to baseline results.

This script compares experiment results to the results reported in the
[paper](https://arxiv.org/pdf/2211.11972.pdf). It takes as input a CSV file
containing experiment results, and outputs a table of p-values comparing the experiment
results to the baseline results.

Usage:
    $ python compare_to_baseline.py <path_to_results_file>

The results file should be a CSV file containing the following columns:
    * algo: The name of the imitation algorithm.
    * env_name: The name of the environment.
    * imit_return_summary: A string containing the mean and standard deviation of the
      experiment returns, as reported by `imitation.scripts.analyze`.
"""

import numpy as np
import pandas as pd
import scipy

from imitation.data import types


def compare_results_to_baseline(results_file: types.AnyPath) -> pd.DataFrame:
    """Compare benchmark results to baseline results.

    Args:
        results_file: Path to a CSV file containing experiment results.

    Returns:
        A string containing a table of p-values comparing the experiment results to
        the baseline results.
    """
    data = pd.read_csv(results_file)
    data["imit_return"] = data["imit_return_summary"].apply(
        lambda x: float(x.split(" ")[0]),
    )
    summary = (
        data[["algo", "env_name", "imit_return"]]
        .groupby(["algo", "env_name"])
        .describe()
    )
    summary.columns = summary.columns.get_level_values(1)
    summary = summary.reset_index()

    # Table 2 (https://arxiv.org/pdf/2211.11972.pdf)
    # todo: store results in this repo outside this file
    baseline = pd.DataFrame.from_records(
        [
            {
                "algo": "??exp_command=bc",
                "env_name": "seals/Ant-v0",
                "mean": 1953,
                "margin": 123,
            },
            {
                "algo": "??exp_command=bc",
                "env_name": "seals/HalfCheetah-v0",
                "mean": 3446,
                "margin": 130,
            },
        ],
    )
    baseline["count"] = 5
    baseline["confidence_level"] = 0.95
    # Back out the standard deviation from the margin of error.
    baseline["std"] = (
        baseline["margin"] * np.sqrt(baseline["count"])
    ) / scipy.stats.t.ppf(
        1 - ((1 - baseline["confidence_level"]) / 2),
        baseline["count"] - 1,
    )

    comparison = pd.merge(summary, baseline, on=["algo", "env_name"])

    comparison["pvalue"] = scipy.stats.ttest_ind_from_stats(
        comparison["mean_x"],
        comparison["std_x"],
        comparison["count_x"],
        comparison["mean_y"],
        comparison["std_y"],
        comparison["count_y"],
    ).pvalue

    return comparison[["algo", "env_name", "pvalue"]]


def main() -> None:  # pragma: no cover
    """Run the script."""
    import sys

    if len(sys.argv) != 2:
        print("Supply a path to a results file")
    else:
        print(compare_results_to_baseline(sys.argv[1]).to_string())


if __name__ == "__main__":
    main()
