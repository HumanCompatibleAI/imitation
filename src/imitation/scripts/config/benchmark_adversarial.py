import math
import multiprocessing
import os.path as osp

import sacred

from imitation.util import util

benchmark_adversarial_ex = sacred.Experiment("benchmark_adversarial")


@benchmark_adversarial_ex.config
def config():
  n_workers = math.ceil(
    multiprocessing.cpu_count() / 4)  # Number of jobs to run simultaneously
  log_root = osp.join("output", "gail_benchmark")
  csv_config_path = "experiments/gail_benchmark_config.csv"  # Config file
  parallel = True  # If True, then use multiple processes. Otherwise threads.


@benchmark_adversarial_ex.config
def paths(log_root):
  log_dir = osp.join(log_root, util.make_unique_timestamp())
  seeds = list(range(3))


@benchmark_adversarial_ex.named_config
def gail():
  extra_named_configs = ["gail"]


@benchmark_adversarial_ex.named_config
def airl():
  extra_named_configs = ["airl"]


@benchmark_adversarial_ex.named_config
def fast():
    """Minimize computation for debugging purposes."""
    n_workers = 1
    seeds = [666]
    extra_named_configs = ["fast"]
    parallel = False  # False may be necessary for proper code coverage.
