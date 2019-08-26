import math
import multiprocessing
import os.path as osp

import sacred

from imitation.util import util

multi_expert_demos_ex = sacred.Experiment("multi_expert_demos")


@multi_expert_demos_ex.config
def config():
  csv_config_path = "FILL ME"  # Config file
  n_workers = math.ceil(
    multiprocessing.cpu_count() / 4)  # Number of jobs to run simultaneously
  log_root = osp.join("output", "multi_expert_demos")
  parallel = True  # If True, then use multiple processes. Otherwise threads.


@multi_expert_demos_ex.config
def paths(log_root):
  log_dir = osp.join(log_root, util.make_unique_timestamp())
  seeds = list(range(3))


@multi_expert_demos_ex.named_config
def fast():
  """Minimize computation for debugging purposes."""
  seeds = [666]
  extra_named_configs = ["fast"]
  parallel = False  # False may be necessary for proper code coverage.
