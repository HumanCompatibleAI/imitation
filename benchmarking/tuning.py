"""Tunes the hyperparameters of the algorithms."""

import copy
import pathlib
from typing import Any, Dict

import numpy as np
import ray
from pandas.api import types as pd_types
from sacred.observers import FileStorageObserver
from tuning_config import parallel_ex, tuning_ex


@tuning_ex.main
def tune(
    parallel_run_config: Dict[str, Any],
    eval_best_trial_resource_multiplier: int = 1,
    num_eval_seeds: int = 5,
) -> None:
    """Tune hyperparameters of imitation algorithms using parallel script.

    Args:
        parallel_run_config: Dictionary of arguments to pass to the parallel script.
        eval_best_trial_resource_multiplier: Factor by which to multiply the
            number of cpus per trial in `resources_per_trial`. This is useful for
            allocating more resources per trial to the evaluation trials than the
            resources for hyperparameter tuning since number of evaluation trials
            is usually much smaller than the number of tuning trials.
        num_eval_seeds: Number of distinct seeds to evaluate the best trial on.
            Set to 0 to disable evaluation.

    Raises:
        ValueError: If no trials are returned by.
    """
    run = parallel_ex.run(config_updates=parallel_run_config)
    experiment_analysis = run.result
    if not experiment_analysis.trials:
        raise ValueError(
            "No trials found. Please ensure that the `experiment_checkpoint_path` "
            "in `parallel_run_config` is passed correctly "
            "or that the tuning run finished properly.",
        )

    return_key = "imit_stats/monitor_return_mean"
    if parallel_run_config["sacred_ex_name"] == "train_rl":
        return_key = "monitor_return_mean"
    best_trial = find_best_trial(experiment_analysis, return_key, print_return=True)

    if num_eval_seeds > 0:  # evaluate the best trial
        resources_per_trial_eval = copy.deepcopy(
            parallel_run_config["resources_per_trial"],
        )
        # update cpus per trial only if it is provided in `resources_per_trial`
        # Uses the default values (cpu=1) if it is not provided
        if "cpu" in parallel_run_config["resources_per_trial"]:
            resources_per_trial_eval["cpu"] *= eval_best_trial_resource_multiplier
        evaluate_best_trial(
            best_trial,
            num_eval_seeds,
            parallel_run_config,
            resources_per_trial_eval,
            return_key,
        )


def find_best_trial(
    experiment_analysis: ray.tune.analysis.ExperimentAnalysis,
    return_key: str,
    print_return: bool = False,
) -> ray.tune.experiment.Trial:
    """Find the trial with the best mean return across all seeds.

    Args:
        experiment_analysis: The result of a parallel/tuning experiment.
        return_key: The key of the return metric in the results dataframe.
        print_return: Whether to print the mean and std of the returns
            of the best trial.

    Returns:
        best_trial: The trial with the best mean return across all seeds.
    """
    df = experiment_analysis.results_df
    # convert object dtype to str required by df.groupby
    for col in df.columns:
        if pd_types.is_object_dtype(df[col]):
            df[col] = df[col].astype("str")
    # group into separate HP configs
    grp_keys = [c for c in df.columns if c.startswith("config") and "seed" not in c]
    grps = df.groupby(grp_keys)
    # store mean return of runs across all seeds in a group
    df["mean_return"] = grps[return_key].transform(lambda x: x.mean())
    best_config_df = df[df["mean_return"] == df["mean_return"].max()]
    row = best_config_df.iloc[0]
    best_config_tag = row["experiment_tag"]
    assert experiment_analysis.trials is not None  # for mypy
    best_trial = [
        t for t in experiment_analysis.trials if best_config_tag in t.experiment_tag
    ][0]

    if print_return:
        all_returns = df[df["mean_return"] == row["mean_return"]][return_key]
        all_returns = all_returns.to_numpy()
        print("All returns:", all_returns)
        print("Mean return:", row["mean_return"])
        print("Std return:", np.std(all_returns))
        print("Total seeds:", len(all_returns))
    return best_trial


def evaluate_best_trial(
    best_trial: ray.tune.experiment.Trial,
    num_eval_seeds: int,
    parallel_run_config: Dict[str, Any],
    resources_per_trial: Dict[str, int],
    return_key: str,
    print_return: bool = False,
):
    """Evaluate the best trial of a parallel run on a separate set of seeds.

    Args:
        best_trial: The trial with the best mean return across all seeds.
        num_eval_seeds: Number of distinct seeds to evaluate the best trial on.
        parallel_run_config: Dictionary of arguments passed to the parallel
            script to get best_trial.
        resources_per_trial: Resources to be used for each evaluation trial.
        return_key: The key of the return metric in the results dataframe.
        print_return: Whether to print the mean and std of the evaluation returns.

    Returns:
        eval_run: The result of the evaluation run.
    """
    best_config = best_trial.config
    best_config["config_updates"].update(
        seed=ray.tune.grid_search(list(range(100, 100 + num_eval_seeds))),
    )
    eval_config_updates = parallel_run_config.copy()
    eval_config_updates.update(
        run_name=parallel_run_config["run_name"] + "_best_hp_eval",
        num_samples=1,
        search_space=best_config,
        resources_per_trial=resources_per_trial,
        search_alg=None,
        repeat=1,
        experiment_checkpoint_path="",
    )
    eval_run = parallel_ex.run(config_updates=eval_config_updates)
    eval_result = eval_run.result
    returns = eval_result.results_df[return_key].to_numpy()
    if print_return:
        print("All returns:", returns)
        print("Mean:", np.mean(returns))
        print("Std:", np.std(returns))
    return eval_run


def main_console():
    observer_path = pathlib.Path.cwd() / "output" / "sacred" / "tuning"
    observer = FileStorageObserver(observer_path)
    tuning_ex.observers.append(observer)
    tuning_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
