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
  base_named_configs = []  # Background settings before search_space is applied
  base_config_updates = {}  # Background settings before search_space is applied
  search_space = {
    "named_configs": [],
    "config_updates": {},
  }  # `config` argument to `ray.tune.run(trainable, config)`
  s3_bucket = None  # Used to create default `upload_dir` if not None.
  upload_dir_uuid_prefix = None  # Optional prefix for timestamp directory


@parallel_ex.config
def ray_upload_dir(inner_experiment_name, s3_bucket, upload_dir_uuid_prefix):
  if s3_bucket is not None:
    uuid = util.make_unique_timestamp()
    if upload_dir_uuid_prefix is not None:
      uuid = f"{upload_dir_uuid_prefix}_{uuid}"
    upload_dir = "s3://{}".format(
      osp.join(s3_bucket, inner_experiment_name, uuid))
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
  """Example config that spins up 8 different training runs of cartpole."""
  inner_experiment_name = "expert_demos"
  search_space = {
    "config_updates": {
      "seed": tune.grid_search([0, 1]),
      "init_rl_kwargs": {
        "learning_rate": tune.grid_search([3e-4, 2e-4]),
        "nminibatches": tune.grid_search([16, 32]),
      },
    }}
  base_named_configs = ["cartpole"]
  base_config_updates = {"init_tensorboard": True}
