import math
import multiprocessing
import os.path as osp

import sacred

from imitation.util import util

multi_train_ex = sacred.Experiment("multi_train_adversarial")


@multi_train_ex.config
def config():
  n_workers = math.ceil(
    multiprocessing.cpu_count() / 4)  # Number of jobs to run simultaneously
  log_root = osp.join("output", "multi_train_adversarial")
  csv_config_path = "experiments/gail_benchmark_config.csv"  # Config file
  parallel = True  # If True, then use multiple processes. Otherwise threads.


@multi_train_ex.config
def paths(log_root):
  log_dir = osp.join(log_root, util.make_unique_timestamp())
  seeds = list(range(3))


@multi_train_ex.named_config
def gail():
  extra_named_configs = ["gail"]


@multi_train_ex.named_config
def airl():
  extra_named_configs = ["airl"]


@multi_train_ex.named_config
def fast():
  """Minimize computation for debugging purposes."""
  seeds = [666]
  extra_named_configs = ["fast"]
  parallel = False  # False may be necessary for proper code coverage.
