"""Runs a Sacred experiment in parallel."""

import collections.abc
import copy
import glob
import pathlib
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import ray
import ray.tune
import sacred
from pandas.api.types import is_object_dtype
from ray.tune import search
from ray.tune.registry import register_trainable
from ray.tune.search import optuna
from sacred.observers import FileStorageObserver

from imitation.scripts.config.parallel import parallel_ex


@parallel_ex.main
def parallel(
    sacred_ex_name: str,
    run_name: str,
    num_samples: int,
    search_space: Mapping[str, Any],
    base_named_configs: Sequence[str],
    base_config_updates: Mapping[str, Any],
    resources_per_trial: Dict[str, Any],
    init_kwargs: Mapping[str, Any],
    local_dir: Optional[str],
    upload_dir: Optional[str],
    repeat: int = 1,
    eval_best_trial: bool = False,
    eval_best_trial_resource_multiplier: int = 1,
    eval_trial_seeds: int = 5,
    experiment_checkpoint_path: str = "",
    syncer=None,
    resume: Union[str, bool] = False,
) -> None:
    """Parallelize multiple runs of another Sacred Experiment using Ray Tune.

    A Sacred FileObserver is attached to the inner experiment and writes Sacred
    logs to "{RAY_LOCAL_DIR}/sacred/". These files are automatically copied over
    to `upload_dir` if that argument is provided.

    Args:
        sacred_ex_name: The Sacred experiment to tune. Either "train_rl" or
            "train_adversarial".
        run_name: A name describing this parallelizing experiment.
            This argument is also passed to `ray.tune.run` as the `name` argument.
            It is also saved in 'sacred/run.json' of each inner Sacred experiment
            under the 'experiment.name' key. This is equivalent to using the Sacred
            CLI '--name' option on the inner experiment. Offline analysis jobs can use
            this argument to group similar data.
        num_samples: Number of times to sample from the hyperparameter space without
            considering repetition using `repeat`.
        search_space: A dictionary which can contain Ray Tune search objects like
            `ray.tune.grid_search` and `ray.tune.sample_from`, and is
            passed as the `config` argument to `ray.tune.run()`. After the
            `search_space` is transformed by Ray, it passed into
            `sacred_ex.run(**run_kwargs)` as `run_kwargs` (`sacred_ex` is the Sacred
            Experiment selected via `sacred_ex_name`). Usually `search_space` only has
            the keys "named_configs" and "config_updates", but any parameter names
            to `sacred.Experiment.run()` are okay.
        base_named_configs: Default Sacred named configs. Any named configs
            taken from `search_space` are higher priority than the base_named_configs.
            Concretely, this priority is implemented by appending named configs taken
            from `search_space` to the run's named configs after `base_named_configs`.
            Named configs in `base_named_configs` don't appear in the automatically
            generated Ray directory name, unlike named configs from `search_space`.
        base_config_updates: Default Sacred config updates. Any config updates taken
            from `search_space` are higher priority than `base_config_updates`.
            Config updates in `base_config_updates` don't appear in the automatically
            generated Ray directory name, unlike config updates from `search_space`.
        resources_per_trial: Argument to `ray.tune.run()`.
        init_kwargs: Arguments to pass to `ray.init`.
        local_dir: `local_dir` argument to `ray.tune.run()`.
        upload_dir: `upload_dir` argument to `ray.tune.run()`.
        repeat: Number of runs to repeat each trial for.
        eval_best_trial: Whether to evaluate the trial with the best mean return
            at the end of tuning on a separate set of seeds.
        eval_best_trial_resource_multiplier: factor by which to multiply the
            number of cpus per trial in `resources_per_trial`.
        eval_trial_seeds: Number of distinct seeds to evaluate the best trial on.
        experiment_checkpoint_path: Path containing the checkpoints of a previous
            experiment ran using this script. Useful for resuming cancelled trials
            of the experiments (using `resume`) or evaluating the best trial of the
            experiment (using `eval_best_trial`).
        resume: If true and `experiment_checkpoint_path` is given, then resumes the
            experiment by restarting the trials that did not finish in the experiment
            checkpoint path.
        syncer: `syncer` argument to `ray.tune.syncer.SyncConfig`.


    Raises:
        TypeError: Named configs not string sequences or config updates not mappings.
    """
    # Basic validation for config options before we enter parallel jobs.
    if not isinstance(base_named_configs, collections.abc.Sequence):
        raise TypeError("base_named_configs must be a Sequence")

    if not isinstance(base_config_updates, collections.abc.Mapping):
        raise TypeError("base_config_updates must be a Mapping")

    if not isinstance(search_space["named_configs"], collections.abc.Sequence):
        raise TypeError('search_space["named_configs"] must be a Sequence')

    if not isinstance(search_space["config_updates"], collections.abc.Mapping):
        raise TypeError('search_space["config_updates"] must be a Mapping')

    # Convert Sacred's ReadOnlyList to List because not picklable.
    base_named_configs = list(base_named_configs)

    # Convert Sacred's ReadOnlyDict (and recursively convert ReadOnlyContainer values)
    # to regular python variants because not picklable.
    base_config_updates = copy.deepcopy(base_config_updates)
    search_space = copy.deepcopy(search_space)

    trainable = _ray_tune_sacred_wrapper(
        sacred_ex_name,
        run_name,
        base_named_configs,
        base_config_updates,
    )

    ray.init(**init_kwargs)
    search_alg = search.Repeater(optuna.OptunaSearch(), repeat=repeat)

    if sacred_ex_name == "train_rl":
        return_key = "monitor_return_mean"
    else:
        return_key = "imit_stats/monitor_return_mean"

    try:
        if experiment_checkpoint_path:
            if resume:
                # restart failed runs from experiment_checkpoint_path
                register_trainable("inner", trainable)
                runner = ray.tune.execution.trial_runner.TrialRunner(
                    local_checkpoint_dir=experiment_checkpoint_path,
                    sync_config=ray.tune.syncer.SyncConfig(
                        upload_dir=upload_dir,
                        syncer=syncer,
                    ),
                    metric=return_key,
                    resume=True,
                )
                print(
                    "Live trials:",
                    len(runner._live_trials),
                    "/",
                    len(runner._trials),
                )
                while not runner.is_finished():
                    runner.step()
                    print("Debug:", runner.debug_string())

            # load experiment analysis results
            result = ray.tune.ExperimentAnalysis(experiment_checkpoint_path)
            result._load_checkpoints_from_latest(
                glob.glob(experiment_checkpoint_path + "/experiment_state*.json"),
            )
            # update result.trials using all the experiment_state json files
            result.trials = None
            result.fetch_trial_dataframes()
        else:
            # run hyperparameter tuning
            result = ray.tune.run(
                trainable,
                config=search_space,
                num_samples=num_samples * repeat,
                name=run_name,
                local_dir=local_dir,
                resources_per_trial=resources_per_trial,
                sync_config=ray.tune.syncer.SyncConfig(
                    upload_dir=upload_dir,
                    syncer=syncer,
                ),
                search_alg=search_alg,
                metric=return_key,
                mode="max",
            )
        if eval_best_trial:
            df = result.results_df
            df = df[df["config/named_configs"].notna()]
            # convert object dtype to str required by df.groupby
            for col in df.columns:
                if is_object_dtype(df[col]):
                    df[col] = df[col].astype("str")
            # group into separate HP configs
            grp_keys = [
                c for c in df.columns if c.startswith("config") and "seed" not in c
            ]
            grps = df.groupby(grp_keys)
            # store mean return of runs across all seeds in a group
            df["mean_return"] = grps[return_key].transform(lambda x: x.mean())
            best_config_df = df[df["mean_return"] == df["mean_return"].max()]
            row = best_config_df.iloc[0]
            best_config_tag = row["experiment_tag"]
            if result.trials is not None:
                trial = [
                    t for t in result.trials if best_config_tag in t.experiment_tag
                ][0]
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

                resources_per_trial_eval = copy.deepcopy(resources_per_trial)
                # update cpus per trial only if it is provided in `resources_per_trial`
                # Uses the default values (cpu=1) if it is not provided
                if "cpu" in resources_per_trial:

                    resources_per_trial_eval[
                        "cpu"
                    ] *= eval_best_trial_resource_multiplier
                    best_config["config_updates"].update(
                        environment=dict(num_vec=resources_per_trial_eval["cpu"]),
                    )

                eval_result = ray.tune.run(
                    trainable,
                    config={
                        "named_configs": best_config["named_configs"],
                        "config_updates": best_config["config_updates"],
                        "command_name": best_config.get("command_name", None),
                    },
                    name=run_name + "_best_hp_eval",
                    resources_per_trial=resources_per_trial_eval,
                )
                returns = eval_result.results_df[return_key].to_numpy()
                print("All returns:", returns)
                print("Mean:", np.mean(returns))
                print("Std:", np.std(returns))
    finally:
        ray.shutdown()


def _ray_tune_sacred_wrapper(
    sacred_ex_name: str,
    run_name: str,
    base_named_configs: list,
    base_config_updates: Mapping[str, Any],
) -> Callable[[Dict[str, Any], Any], Mapping[str, Any]]:
    """From an Experiment build a wrapped run function suitable for Ray Tune.

    `ray.tune.run(...)` expects a trainable function that takes a dict
    argument `config`. The wrapped function uses `config` as keyword args for
    `ex.run(...)` because we want to be able to hyperparameter tune over both the
    `named_configs` and `config_updates` arguments.

    The Ray Tune `reporter` is not passed to the inner experiment.

    Args:
        sacred_ex_name: The Sacred experiment to tune. Either "train_rl" or
            "train_adversarial".
        run_name: A name describing this parallelizing experiment.
            This argument is also passed to `ray.tune.run` as the `name` argument.
            It is also saved in 'sacred/run.json' of each inner Sacred experiment
            under the 'experiment.name' key. This is equivalent to using the Sacred
            CLI '--name' option on the inner experiment. Offline analysis jobs can use
            this argument to group similar data.
        base_named_configs: Default Sacred named configs. Any named configs
            taken from `search_space` are higher priority than the base_named_configs.
            Concretely, this priority is implemented by appending named configs taken
            from `search_space` to the run's named configs after `base_named_configs`.
            Named configs in `base_named_configs` don't appear in the automatically
            generated Ray directory name, unlike named configs from `search_space`.
        base_config_updates: Default Sacred config updates. Any config updates taken
            from `search_space` are higher priority than `base_config_updates`.
            Config updates in `base_config_updates` don't appear in the automatically
            generated Ray directory name, unlike config updates from `search_space`.

    Returns:
        A function that takes two arguments, `config` (used as keyword args for
        `ex.run`) and `reporter`. The function returns the run result.
    """

    def inner(config: Dict[str, Any], reporter) -> Mapping[str, Any]:
        """Trainable function with the correct signature for `ray.tune`.

        Args:
            config: Keyword arguments for `ex.run()`, where `ex` is the
                `sacred.Experiment` instance associated with `sacred_ex_name`.
            reporter: Callback to report progress to Ray.

        Returns:
            Result from `ray.Run` object.
        """
        # Set Sacred capture mode to "sys" because default "fd" option leads to error.
        # See https://github.com/IDSIA/sacred/issues/289.
        # TODO(shwang): Stop modifying CAPTURE_MODE once the issue is fixed.
        sacred.SETTINGS.CAPTURE_MODE = "sys"

        run_kwargs = config
        updated_run_kwargs: Dict[str, Any] = {}
        # Import inside function rather than in module because Sacred experiments
        # are not picklable, and Ray requires this function to be picklable.
        from imitation.scripts.train_adversarial import train_adversarial_ex
        from imitation.scripts.train_imitation import train_imitation_ex
        from imitation.scripts.train_preference_comparisons import (
            train_preference_comparisons_ex,
        )
        from imitation.scripts.train_rl import train_rl_ex

        experiments = {
            "train_rl": train_rl_ex,
            "train_adversarial": train_adversarial_ex,
            "train_imitation": train_imitation_ex,
            "train_preference_comparisons": train_preference_comparisons_ex,
        }
        ex = experiments[sacred_ex_name]

        # Apply base configs to get modified `named_configs` and `config_updates`.
        named_configs = base_named_configs + run_kwargs["named_configs"]
        updated_run_kwargs["named_configs"] = named_configs

        config_updates: Dict[str, Any] = {}
        config_updates.update(base_config_updates)
        config_updates.update(run_kwargs["config_updates"])
        # for repeat runs, set the seed using their trial index
        if "__trial_index__" in run_kwargs:
            config_updates.update(seed=run_kwargs.pop("__trial_index__"))
        updated_run_kwargs["config_updates"] = config_updates

        # Add other run_kwargs items to updated_run_kwargs.
        for k, v in run_kwargs.items():
            if k not in updated_run_kwargs:
                updated_run_kwargs[k] = v
        run = ex.run(
            **updated_run_kwargs,
            options={"--run": run_name, "--file_storage": "sacred"},
        )

        assert run.status == "COMPLETED"
        return run.result

    return inner


def main_console():
    observer_path = pathlib.Path.cwd() / "output" / "sacred" / "parallel"
    observer = FileStorageObserver(observer_path)
    parallel_ex.observers.append(observer)
    parallel_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
