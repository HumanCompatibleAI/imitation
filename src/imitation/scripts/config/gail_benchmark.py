from collections import OrderedDict
import os.path as osp

import sacred

from imitation import util

gail_benchmark_ex = sacred.Experiment("gail_benchmark")

experiment_specs = [
  ("CartPole-v0", [1, 4, 7, 10]),

  # GAIL paper uses Acrobot-v0, but it doesn't exist in Gym any more.
  ("Acrobot-v1", [1, 4, 7, 10]),
  ("MountainCar-v0", [1, 4, 7, 10]),

  # TODO(shwang): Uncomment these specs after expert demonstrations are
  # available.

  # ("HalfCheetah-v1", [4, 11, 18, 25]),

  # # GAIL paper uses Hopper-v0 but that no longer exists.
  # ("Hopper-v1", [4, 11, 18, 25]),

  # # GAIL paper uses Walker-v1 but that no longer exists.
  # ("Walker2d-v2", [4, 11, 18, 25]),

  # # GAIL paper uses Ant-v2 but that no longer exists.
  # ("Ant-v2", [4, 11, 18, 25]),

  # # GAIL paper uses Humanoid-v1 but that no longer exists.
  # ("Humanoid-v2", [80, 160, 240]),
]
"""Format: (env_name, List[n_demonstrations])"""

# TODO: Part of the GAIL paper, but don't have expert demonstrations available.
# experiments_specs_special = [
#   # GAIL paper uses Reacher-v1, but that no longer exists.
#   ("Reacher-v2", [4, 11, 18], [0, 1e-3, 1e-2])
# ]
# """Format: (env_name, List[n_demonstrations], List[lambda])"""


@gail_benchmark_ex.config
def config():
  config_updates_list = []  # Different train_ex configs to train.

  # Build all permutations of config updates from `experiment_specs`.
  for env_name, n_demos_list in experiment_specs:
    for n_demonstrations in n_demos_list:
      config_updates_list.append(OrderedDict(
        env_name=env_name,
        n_demonstrations=n_demonstrations,
      ))

  log_root = osp.join("output", "gail_benchmark")  # output directory


@gail_benchmark_ex.config
def logging(env_name, log_root):
  log_dir = osp.join(log_root, env_name.replace('/', '_'),
                     util.make_timestamp())

@gail_benchmark_ex.named_config
def fast():
  train_ex_named_configs = ["fast"]
