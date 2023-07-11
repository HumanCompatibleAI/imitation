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
    parallel: Dict[str, Any],
    eval_best_trial: bool = False,
    eval_best_trial_resource_multiplier: int = 1,
    eval_trial_seeds: int = 5,
) -> None:
    """Tune hyperparameters of imitation algorithms using parallel script.

    Args:
        parallel: A dictionary of arguments from the parallel script.
        eval_best_trial: Whether to evaluate the trial with the best mean return
            at the end of tuning on a separate set of seeds.
        eval_best_trial_resource_multiplier: factor by which to multiply the
            number of cpus per trial in `resources_per_trial`.
        eval_trial_seeds: Number of distinct seeds to evaluate the best trial on.
    """
    run = parallel_ex.run(config_updates=parallel)
    result = run.result

    if eval_best_trial:
        if parallel["sacred_ex_name"] == "train_rl":
            return_key = "monitor_return_mean"
        else:
            return_key = "imit_stats/monitor_return_mean"
        df = result.results_df
        df = df[df["config/named_configs"].notna()]
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
        if result.trials is not None:
            trial = [t for t in result.trials if best_config_tag in t.experiment_tag][0]
            best_config = trial.config
            print("Mean return:", row["mean_return"])
            print(
                "All returns:",
                df[df["mean_return"] == row["mean_return"]][return_key],
            )
            print("Total seeds:", (df["mean_return"] == row["mean_return"]).sum())

            best_config["config_updates"].update(
                seed=ray.tune.grid_search(list(range(100, 100 + eval_trial_seeds))),
            )

            resources_per_trial_eval = copy.deepcopy(parallel["resources_per_trial"])
            # update cpus per trial only if it is provided in `resources_per_trial`
            # Uses the default values (cpu=1) if it is not provided
            if "cpu" in parallel["resources_per_trial"]:
                resources_per_trial_eval["cpu"] *= eval_best_trial_resource_multiplier

            eval_config_updates = parallel.copy()
            eval_config_updates.update(
                run_name=parallel["run_name"] + "_best_hp_eval",
                num_samples=1,
                search_space=best_config,
                base_named_configs=parallel["base_named_configs"],
                base_config_updates=parallel["base_config_updates"],
                resources_per_trial=resources_per_trial_eval,
                search_alg=None,
                repeat=1,
                experiment_checkpoint_path="",
                resume=False,
            )
            eval_run = parallel_ex.run(config_updates=eval_config_updates)
            eval_result = eval_run.result
            returns = eval_result.results_df[return_key].to_numpy()
            print("All returns:", returns)
            print("Mean:", np.mean(returns))
            print("Std:", np.std(returns))


def main_console():
    observer_path = pathlib.Path.cwd() / "output" / "sacred" / "tuning"
    observer = FileStorageObserver(observer_path)
    tuning_ex.observers.append(observer)
    tuning_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
