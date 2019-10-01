import os.path as osp
from typing import Callable, Optional

import ray
import ray.tune
from sacred.observers import FileStorageObserver

from imitation.scripts.config.parallel import parallel_ex


@parallel_ex.main
def parallel(inner_experiment_name: str,
             inner_run_name: str,
             search_space: dict,
             base_named_configs: list,
             base_config_updates: dict,
             resources_per_trial: dict,
             local_dir: Optional[str],
             upload_dir: Optional[str],
             ) -> None:
  """Parallelize multiple runs of another Sacred Experiment using Ray Tune.

  A Sacred FileObserver is attached to the inner experiment and writes Sacred
  logs to "{RAY_LOCAL_DIR}/sacred/". These files are automatically copied over
  to `upload_dir` if that argument is provided.

  Args:
    inner_experiment_name: The experiment to tune. Either "expert_demos" or
      "train_adversarial".
    inner_run_name: A name describing this parallelizing experiment.
      Added to each 'sacred/run.json' under the 'experiment.name' key.
      This is equivalent to using the Sacred CLI '--name' option on the
      inner experiment. Offline analysis jobs can use this argument to group
      similar data.
    search_space: `config` argument to `ray.tune.run()`.
    base_named_configs: `search_space["named_configs"]` is appended to this list
      before it is passed to the inner experiment's `run()`. Notably,
      `base_named_configs` doesn't appear in the automatically generated
      Ray directory name.
    base_config_updates: `search_space["config_updates"]` is applied to this
      dict before it is passed to the inner experiment's `run()`.
    resource_per_trial: Argument to `ray.tune.run()`.
    local_dir: `local_dir` argument to `ray.tune.run()`.
    upload_dir: `upload_dir` argument to `ray.tune.run()`.
  """
  trainable = _ray_tune_sacred_wrapper(inner_experiment_name,
                                       inner_run_name,
                                       base_named_configs,
                                       base_config_updates)

  # Disable all Ray Loggers.
  #
  # JSON and CSV loggers are redundant now that we have Sacred logs.
  # TensorBoard logs don't contain useful information (inner experiment never
  # gets access to `reporter`), and clog up the TensorBoard Runs dashboard.
  ray_loggers = ()

  ray.init()
  try:
    ray.tune.run(trainable, config=search_space,
                 local_dir=local_dir, upload_dir=upload_dir,
                 loggers=ray_loggers, resources_per_trial=resources_per_trial)
  finally:
    ray.shutdown()


def _ray_tune_sacred_wrapper(inner_experiment_name: str,
                             inner_run_name: str,
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
    # Import inside function rather than in module because Sacred experiments
    # are not picklable, and Ray requires this function to be picklable.
    from imitation.scripts.expert_demos import expert_demos_ex
    from imitation.scripts.train_adversarial import train_ex
    experiments = {
      "expert_demos": expert_demos_ex,
      "train_adversarial": train_ex,
    }
    ex = experiments[inner_experiment_name]

    observer = FileStorageObserver.create('sacred')
    ex.observers.append(observer)

    # Apply base configs
    base_named_configs.extend(config.get("named_configs", []))
    base_config_updates.update(config.get("config_updates", {}))
    config["named_configs"] = base_named_configs
    config["config_updates"] = base_config_updates

    run = ex.run(**config,
                 options={"--run": inner_run_name})

    # Ray Tune has a string formatting error if raylet completes without
    # any calls to `reporter`.
    reporter(done=True)

    assert run.status == 'COMPLETED'
    return run.result
  return inner


if __name__ == '__main__':
  observer = FileStorageObserver.create(
      osp.join('output', 'sacred', 'parallel'))
  parallel_ex.observers.append(observer)
  parallel_ex.run_commandline()
