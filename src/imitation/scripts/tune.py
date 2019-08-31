import functools
from typing import Callable

import ray
import ray.tune
import sacred

from imitation.scripts.config.tune import tune_ex
from imitation.scripts.expert_demos import expert_demos_ex
from imitation.scripts.train_adversarial import train_ex

experiments = {
  "expert_demos": expert_demos_ex,
  "train_adversarial": train_ex,
}


@tune_ex.main
def tune(inner_experiment_name: str, search_space: dict) -> None:
  """Start a hyperparameter tuning experiment.

  Args:
    inner_experiment_name: The experiment to tune. Either "expert_demos" or
      "train_adversarial".
    search_space: `config` argument to `ray.tune.run(trainable, config)`.
  """
  ray.init()
  trainable = _ray_tune_sacred_wrapper(experiments[inner_experiment_name])
  ray.tune.run(trainable, config=search_space)


def _ray_tune_sacred_wrapper(
  ex: sacred.Experiment,
) -> Callable[[dict], dict]:
  """From an Experiment build a wrapped run function suitable for Ray Tune.

  `ray.tune.run(...)` expects a trainable function that takes a single dict
  argument `config`. The wrapped function uses `config` as keyword args for
  `ex.run(...)` because we want to be able to hyperparameter tune over both the
  `named_configs` and `config_updates` arguments.

  `config["named_configs"]` must contain "ray_tune". (Ensures that Ray Tune
  hooks are enabled in the Experiment run).

  Args:
    ex: The Sacred Experiment.
    command_name: If provided, then run this particular command. Otherwise, run
      the Sacred Experiment's main command.
  Returns:
    A function that takes a single argument, `config` (used as keyword args for
    `ex.run`), and returns the run result.
  """
  @functools.wraps(ex.run)
  def inner(config: dict):
    named_configs = config.get("named_configs", [])
    assert "ray_tune" in named_configs

    run = ex.run(**config)
    assert run.status == 'COMPLETED'
    return run.result
  return inner
