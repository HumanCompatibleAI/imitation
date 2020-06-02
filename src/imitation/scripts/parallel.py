import collections.abc
import copy
import os
from typing import Any, Callable, Dict, Optional

import ray
import ray.tune
from sacred.observers import FileStorageObserver

from imitation.scripts.config.parallel import parallel_ex


@parallel_ex.main
def parallel(
    sacred_ex_name: str,
    run_name: str,
    search_space: dict,
    base_named_configs: list,
    base_config_updates: dict,
    resources_per_trial: dict,
    init_kwargs: Dict[str, Any],
    local_dir: Optional[str],
    upload_dir: Optional[str],
) -> None:
    """Parallelize multiple runs of another Sacred Experiment using Ray Tune.

    A Sacred FileObserver is attached to the inner experiment and writes Sacred
    logs to "{RAY_LOCAL_DIR}/sacred/". These files are automatically copied over
    to `upload_dir` if that argument is provided.

    Args:
      sacred_ex_name: The Sacred experiment to tune. Either "expert_demos" or
        "train_adversarial".
      run_name: A name describing this parallelizing experiment.
        This argument is also passed to `ray.tune.run` as the `name` argument.
        It is also saved in 'sacred/run.json' of each inner Sacred experiment
        under the 'experiment.name' key. This is equivalent to using the Sacred
        CLI '--name' option on the inner experiment. Offline analysis jobs can use
        this argument to group similar data.
      search_space: `config` argument to `ray.tune.run()`.
      base_named_configs: `search_space["named_configs"]` is appended to this list
        before it is passed to the inner experiment's `run()`. Notably,
        `base_named_configs` doesn't appear in the automatically generated
        Ray directory name.
      base_config_updates: `search_space["config_updates"]` is applied to this
        dict before it is passed to the inner experiment's `run()`.
      resources_per_trial: Argument to `ray.tune.run()`.
      init_kwargs: Arguments to pass to `ray.init`.
      local_dir: `local_dir` argument to `ray.tune.run()`.
      upload_dir: `upload_dir` argument to `ray.tune.run()`.
    """
    # Basic validation for config options before we enter parallel jobs.
    for name in base_named_configs:
        assert isinstance(name, str)
    for k in base_config_updates:
        assert isinstance(k, str)
    assert isinstance(search_space["named_configs"], collections.abc.Sequence)
    assert isinstance(search_space["config_updates"], collections.abc.Mapping)

    # Explicitly set `data_dir` if parallelizing `train_adversarial`.
    # We need this to automatically find rollout pickles because Ray
    # sets a new working directory for each Raylet.
    if sacred_ex_name == "train_adversarial":
        if "data_dir" not in base_config_updates:
            data_dir = os.path.join(os.getcwd(), "data/")
            base_config_update = dict(base_config_updates)
            base_config_update["data_dir"] = data_dir

    trainable = _ray_tune_sacred_wrapper(
        sacred_ex_name,
        run_name,
        copy.deepcopy(base_named_configs),
        copy.deepcopy(base_config_updates),
    )

    # Disable all Ray Loggers.
    #
    # JSON and CSV loggers are redundant now that we have Sacred logs.
    # TensorBoard logs don't contain useful information (inner Sacred experiment
    # never gets access to `reporter`), and clog up the TensorBoard Runs
    # dashboard.
    ray_loggers = ()

    ray.init(**init_kwargs)
    try:
        ray.tune.run(
            trainable,
            config=copy.deepcopy(search_space),
            name=run_name,
            local_dir=local_dir,
            upload_dir=upload_dir,
            loggers=ray_loggers,
            resources_per_trial=resources_per_trial,
        )
    finally:
        ray.shutdown()


def _ray_tune_sacred_wrapper(
    sacred_ex_name: str,
    run_name: str,
    base_named_configs: list,
    base_config_updates: dict,
) -> Callable:
    """From an Experiment build a wrapped run function suitable for Ray Tune.

    `ray.tune.run(...)` expects a trainable function that takes a dict
    argument `config`. The wrapped function uses `config` as keyword args for
    `ex.run(...)` because we want to be able to hyperparameter tune over both the
    `named_configs` and `config_updates` arguments.

    The Ray Tune `reporter` is not passed to the inner experiment.

    Args have the same meanings as arguments described in `parallel`.

    Returns:
      A function that takes two arguments, `config` (used as keyword args for
      `ex.run`) and `reporter`. The function returns the run result.
    """

    def inner(config: dict, reporter) -> dict:
        """Trainable function with the correct signature for `ray.tune`.

        Args:
            config: Keyword arguments for `ex.run()`, where `ex` is the
                `sacred.Experiment` instance associated with `sacred_ex_name`.
        """
        run_kwargs = config
        updated_run_kwargs = {}
        # Import inside function rather than in module because Sacred experiments
        # are not picklable, and Ray requires this function to be picklable.
        from imitation.scripts.expert_demos import expert_demos_ex
        from imitation.scripts.train_adversarial import train_ex

        experiments = {
            "expert_demos": expert_demos_ex,
            "train_adversarial": train_ex,
        }
        ex = experiments[sacred_ex_name]

        observer = FileStorageObserver("sacred")
        ex.observers.append(observer)

        # Apply base configs to get modified `named_configs` and `config_updates`.
        named_configs = []
        named_configs.extend(base_named_configs)
        named_configs.extend(run_kwargs["named_configs"])
        updated_run_kwargs["named_configs"] = named_configs

        config_updates = {}
        config_updates.update(base_config_updates)
        config_updates.update(run_kwargs["config_updates"])
        updated_run_kwargs["config_updates"] = config_updates

        # Apply
        for k, v in run_kwargs.items():
            if k not in updated_run_kwargs:
                updated_run_kwargs[k] = v

        run = ex.run(**updated_run_kwargs, options={"--run": run_name})

        # Ray Tune has a string formatting error if raylet completes without
        # any calls to `reporter`.
        reporter(done=True)

        assert run.status == "COMPLETED"
        return run.result

    return inner


def main_console():
    observer = FileStorageObserver(os.path.join("output", "sacred", "parallel"))
    parallel_ex.observers.append(observer)
    parallel_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
