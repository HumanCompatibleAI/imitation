"""Config files for hyperparameter tuning experiments.

Hyperparameter experiments are intended to be defined in Python rather than
via CLI. For example, a user should add a new
`@tune_ex.named_config` to define a new hyperparameter experiment.

Adding custom named configs is necessary because the CLI interface can't add
search spaces to the config like `"seed": tune.grid_search([0, 1, 2, 3])`.
"""

import ray.tune as tune
import sacred

tune_ex = sacred.Experiment("tune")


@tune_ex.config
def config():
  inner_experiment_name = "expert_demos"  # The experiment to tune
  search_space = {
    "named_configs": [],
    "config_updates": {},
  }  # `config` argument to `ray.tune.run(trainable, config)`


# Debug named configs
@tune_ex.named_config
def debug_log_root():
  search_space = {"config_updates": {"log_root": "/tmp/debug_tune_ex"}}


# Each named config that follows describes a hyperparameter tuning experiments.

@tune_ex.named_config
def example_cartpole_rl():
  """Not an actual hyperparameter tuning experiment. Just a prototype."""
  inner_experiment_name = "expert_demos"
  search_space = {
    "named_configs": ["cartpole"],
    "config_updates": {
      "seed": tune.grid_search([0, 1, 2, 3]),
      "make_blank_policy_kwargs": {
        "learning_rate": tune.grid_search([3e-4, 2e-4, 1e-4]),
        "nminibatches": tune.grid_search([16, 32, 64]),
      },
    }}
