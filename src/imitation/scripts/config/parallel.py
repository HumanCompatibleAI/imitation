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
  search_space = {
    "named_configs": [],
    "config_updates": {},
  }  # `config` argument to `ray.tune.run(trainable, config)`
  s3_bucket = None  # Used to create default `upload_dir` if not None.


@parallel_ex.config
def ray_upload_dir(inner_experiment_name, s3_bucket):
  if s3_bucket is not None:
    upload_dir = "s3://{}".format(
      osp.join(s3_bucket, inner_experiment_name, util.make_unique_timestamp()))
  else:
    upload_dir = None  # `upload_dir` param from `ray.tune.run`


@parallel_ex.named_config
def s3():
  s3_bucket = "shwang-chai"


# Debug named configs
@parallel_ex.named_config
def debug_log_root():
  search_space = {"config_updates": {"log_root": "/tmp/debug_parallel_ex"}}


@parallel_ex.named_config
def example_cartpole_rl():
  """Example config that spins up 4*4*3 different training runs of cartpole."""
  inner_experiment_name = "expert_demos"
  search_space = {
    "named_configs": ["cartpole", "fast"],
    "config_updates": {
      "seed": tune.grid_search([0, 1, 2, 3]),
      "make_blank_policy_kwargs": {
        "learning_rate": tune.grid_search([3e-4, 2e-4, 1e-4]),
        "nminibatches": tune.grid_search([16, 32, 64]),
      },
    }}
