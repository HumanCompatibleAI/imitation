"""Config files for parallel experiments.

Parallel experiments are intended to be defined in Python rather than
via CLI. For example, a user should add a new
`@parallel_ex.named_config` to define a new parallel experiment.

Adding custom named configs is necessary because the CLI interface can't add
search spaces to the config like `"seed": tune.grid_search([0, 1, 2, 3])`.
"""

import os.path as osp

import ray.tune as tune
import sacred

import imitation.util as util

parallel_ex = sacred.Experiment("parallel")


@parallel_ex.config
def config():
  inner_experiment_name = "expert_demos"  # The experiment to parallelize
  inner_run_name = "DEFAULT"  # CLI --name option. Used for analysis grouping.
  resources_per_trial = {}  # Argument to `tune.run`
  base_named_configs = []  # Background settings before search_space is applied
  base_config_updates = {}  # Background settings before search_space is applied
  search_space = {
    "named_configs": [],
    "config_updates": {},
  }  # `config` argument to `ray.tune.run(trainable, config)`

  upload_dir = None  # `upload_dir` arg for `ray.tune.run`
  n_seeds = 3  # Number of seeds to search over by default


@parallel_ex.config
def seeds(n_seeds):
  search_space = {"config_updates": {
    "seed": tune.grid_search(list(range(n_seeds)))}}


@parallel_ex.named_config
def s3():
  upload_dir = "s3://shwang-chai/private"


# Debug named configs
@parallel_ex.named_config
def debug_log_root():
  search_space = {"config_updates": {"log_root": "/tmp/debug_parallel_ex"}}


@parallel_ex.named_config
def example_cartpole_rl():
  inner_experiment_name = "expert_demos"
  outer_experiment_name = "example-cartpole"
  n_seeds = 2
  search_space = {
    "config_updates": {
      "init_rl_kwargs": {
        "learning_rate": tune.grid_search([3e-4 * x for x in (1/3, 1/2, 1, 2)]),
        "nminibatches": tune.grid_search([16, 32, 64]),
      },
    }}
  base_named_configs = ["cartpole"]
  base_config_updates = {"init_tensorboard": True}
  resources_per_trial=dict(cpu=4)


EASY_ENVS = ["cartpole", "pendulum", "mountain_car"]

@parallel_ex.named_config
def example_rl_easy():
  inner_experiment_name = "expert_demos"
  outer_experiment_name = "example-rl-easy"
  n_seeds = 2
  search_space = {
    "named_configs": tune.grid_search([[env] for env in EASY_ENVS]),
    "config_updates": {
      "init_rl_kwargs": {
        "learning_rate": tune.grid_search([3e-4 * x for x in (1/3, 1/2, 1, 2)]),
        "nminibatches": tune.grid_search([16, 32, 64]),
      },
    }}
  base_config_updates = {"init_tensorboard": True}
  resources_per_trial=dict(cpu=4)


@parallel_ex.named_config
def example_gail_easy():
  inner_experiment_name = "train_adversarial"
  outer_experiment_name = "example-gail-easy"
  n_seeds = 1
  search_space = {
    "named_configs": tune.grid_search([[env] for env in EASY_ENVS]),
    "config_updates": {
      "init_trainer_kwargs": {
        "init_rl_kwargs": {
          "learning_rate": tune.grid_search(
            [3e-4 * x for x in (1/3, 1/2, 1, 2)]),
          "nminibatches": tune.grid_search([16, 32, 64]),
        },
      },
    }}
  base_config_updates = {"init_tensorboard": True,
                         "init_trainer_kwargs": {"use_gail": True},
                         }
