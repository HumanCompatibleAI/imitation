"""Compute the probability that one algorithm improved over another."""
import argparse
import pathlib
import sys
import warnings
from typing import Dict, List, Optional

import numpy as np
from rliable import library as rly
from rliable import metrics

from imitation.util.sacred_file_parsing import SacredRun, group_runs_by_algo_and_env


def sample_matrix_from_runs_by_env(
    runs_by_env: Dict[str, List[SacredRun]],
    envs: Optional[List[str]] = None,
) -> np.ndarray:
    """Samples a matrix of scores from the runs for each environment.

    Note: when the number of samples for each environment is not equal, the samples
    will be truncated to the minimum sample count.

    Args:
        runs_by_env: A dictionary mapping environment names to lists of runs.
        envs: The environments to sample from. If None, all environments are used.

    Returns:
        A matrix of scores of shape (n_samples, n_envs).
    """
    if envs is None:
        envs = list(runs_by_env.keys())

    sample_counts_by_env = {env: len(runs_by_env[env]) for env in envs}

    min_sample_count = min(sample_counts_by_env.values())
    if not all(
        sample_counts_by_env[env] == sample_counts_by_env[envs[0]] for env in envs
    ):
        warnings.warn(
            f"The runs for the environments have different sample counts "
            f"{sample_counts_by_env}. "
            f"This is not supported by the probability of improvement. Therefore, "
            f"samples will be truncated to the minimum sample count of"
            f" {min_sample_count}",
        )

    return np.asarray(
        [
            [
                run["result"]["imit_stats"]["monitor_return_mean"]
                for run in runs_by_env[env][:min_sample_count]
            ]
            for env in envs
        ],
    ).T


def compute_probability_of_improvement(
    runs_by_env: Dict[str, List[SacredRun]],
    baseline_runs_by_env: Dict[str, List[SacredRun]],
    reps: int,
):
    """Computes the probability of improvement of the runs over the baseline runs.

    Args:
        runs_by_env: A dictionary mapping environment names to lists of runs.
        baseline_runs_by_env: A dictionary mapping environment names to lists of runs.
        reps: The number of bootstrap repetitions to use to compute the confidence
            interval.

    Returns:
        A tuple of:
         - probability of improvement
         - confidence interval
         - number of samples per env
         - number of baseline samples per env
    """
    envs = runs_by_env.keys()
    baseline_envs = baseline_runs_by_env.keys()
    comparison_envs = sorted(set(envs).intersection(set(baseline_envs)))

    # We need to arrange the scores in a matrix of shape (n_samples, n_envs).
    # When we do not have the same number of samples for each environment,
    # the samples are truncated to the minimum sample count.
    run_scores = sample_matrix_from_runs_by_env(runs_by_env, comparison_envs)
    assert run_scores.shape[0] >= 1
    assert run_scores.shape[1] == len(comparison_envs)
    baseline_run_scores = sample_matrix_from_runs_by_env(
        baseline_runs_by_env,
        comparison_envs,
    )
    assert baseline_run_scores.shape[0] >= 1
    assert baseline_run_scores.shape[1] == len(comparison_envs)

    samples_per_env = run_scores.shape[0]
    baseline_samples_per_env = baseline_run_scores.shape[0]

    probabs, error_intervals = rly.get_interval_estimates(
        {"baseline_vs_new": (baseline_run_scores, run_scores)},
        metrics.probability_of_improvement,
        reps=reps,
    )
    probability_of_improvement = probabs["baseline_vs_new"]
    confidence_interval = np.squeeze(error_intervals["baseline_vs_new"])

    return (
        probability_of_improvement,
        confidence_interval,
        samples_per_env,
        baseline_samples_per_env,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=pathlib.Path)
    parser.add_argument("baseline_runs_dir", nargs="?", default=None, type=pathlib.Path)
    parser.add_argument("--baseline-algo", type=str)
    parser.add_argument("--algo", type=str)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)

    args = parser.parse_args()

    if args.baseline_runs_dir is None:
        args.baseline_runs_dir = args.runs_dir

    runs_by_algo_and_env = group_runs_by_algo_and_env(
        args.runs_dir,
        only_completed_runs=True,
    )
    baseline_runs_by_algo_and_env = group_runs_by_algo_and_env(
        args.baseline_runs_dir,
        only_completed_runs=True,
    )

    algos = sorted(runs_by_algo_and_env.keys())
    baseline_algos = sorted(baseline_runs_by_algo_and_env.keys())

    try:
        if len(algos) == 0:
            raise ValueError(f"The run directory [{args.runs_dir}] contains no runs.")

        if len(baseline_algos) == 0:
            raise ValueError(
                f"The baseline run directory [{args.baseline_runs_dir}] "
                f"contains no runs.",
            )

        if "algo" not in args is None:
            if len(algos) == 1:
                args.algo = algos[0]
            else:
                raise ValueError(
                    f"The run directory [{args.runs_dir}] contains runs for the "
                    f"algorithms [{', '.join(algos)}]. Please use the --algo option "
                    f" to specify which algorithms runs to compare.",
                )

        if args.baseline_algo is None:
            if len(baseline_algos) == 1:
                args.baseline_algo = baseline_algos[0]
            elif args.algo in baseline_algos:
                args.baseline_algo = args.algo
            else:
                raise ValueError(
                    f"The baseline run directory [{args.baseline_runs_dir}] contains "
                    f"runs for the algorithms [{', '.join(baseline_algos)}]. "
                    f"Please use the --baseline-algo option specify which one to "
                    f"compare to.",
                )

        if args.algo not in algos:
            raise ValueError(
                f"The run directory [{args.runs_dir}] contains runs for the algorithms"
                f" [{', '.join(algos)}]. You specified [{args.algo}], for which no"
                f" runs can be found in the run directory",
            )

        if args.baseline_algo not in baseline_algos:
            raise ValueError(
                f"The baseline run directory [{args.baseline_runs_dir}] contains runs "
                f"for the algorithms [{', '.join(baseline_algos)}]. "
                f"You specified [{args.baseline_algo}], for which no runs can be found"
                f" in the baseline run directory",
            )

        if (args.algo == args.baseline_algo) and (
            args.runs_dir == args.baseline_runs_dir
        ):
            warnings.warn(
                "You are comparing two equal sets of runs. "
                "This is probably not what you want.",
            )

        envs = runs_by_algo_and_env[args.algo].keys()
        baseline_envs = baseline_runs_by_algo_and_env[args.baseline_algo].keys()

        comparison_envs = set(envs).intersection(set(baseline_envs))

        if len(comparison_envs) == 0:
            raise ValueError(
                f"The baseline runs are for the environments "
                f"[{', '.join(baseline_envs)}], while the runs are for the "
                f"environments [{', '.join(envs)}]. "
                f"There is no overlap in the environments of the two run sets, so no "
                f"comparison can be made",
            )

        ignoring_some_envs = len(comparison_envs) < len(envs)
        ignoring_some_baseline_envs = len(comparison_envs) < len(baseline_envs)
        if ignoring_some_envs or ignoring_some_baseline_envs:
            warnings.warn(
                f"The baseline runs are for the environments "
                f"[{', '.join(baseline_envs)}], "
                f"while the runs are for the environments [{', '.join(envs)}]. "
                f"The comparison will only be made for the environments "
                f"[{', '.join(comparison_envs)}].",
            )

    except ValueError as e:
        print(e)
        sys.exit(1)

    (
        probability_of_improvement,
        error_interval,
        n_samples,
        n_baseline_samples,
    ) = compute_probability_of_improvement(
        runs_by_env=runs_by_algo_and_env[args.algo],
        baseline_runs_by_env=baseline_runs_by_algo_and_env[args.baseline_algo],
        reps=args.bootstrap_reps,
    )

    show_path = args.algo == args.baseline_algo
    algo_str = f"{args.algo} ({args.runs_dir})" if show_path else args.algo
    baseline_algo_str = (
        f"{args.baseline_algo} ({args.baseline_runs_dir})"
        if show_path
        else args.baseline_algo
    )

    print(
        f"Comparison based on {n_samples} samples per environment for {algo_str} and"
        f" {n_baseline_samples} samples per environment for {baseline_algo_str}.",
    )
    print(f"Samples taken in {', '.join(comparison_envs)}")
    print()
    print(f"Probability of improvement of {algo_str} over {baseline_algo_str}:")
    print(
        f"{probability_of_improvement:.3f} "
        f"({error_interval[0]:.3f}, {error_interval[1]:.3f}, "
        f"reps={args.bootstrap_reps})",
    )


if __name__ == "__main__":
    main()
