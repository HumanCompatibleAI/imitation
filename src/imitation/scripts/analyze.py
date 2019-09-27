from collections import OrderedDict
import os.path as osp
from typing import Optional

import pandas as pd
from sacred.observers import FileStorageObserver

from imitation.scripts.config.analyze import analysis_ex
import imitation.util.sacred as sacred_util
from imitation.util.sacred import dict_get_nested as get


@analysis_ex.main
def analyze_imitation(source_dir: str,
                      run_name: Optional[str],
                      skip_failed_runs: bool,
                      csv_output_path: Optional[str],
                      verbose: bool,
                      ) -> pd.DataFrame:
  """
  Args:
    source_dir: A directory containing Sacred FileObserver subdirectories
      associated with the `train_adversarial` Sacred script. Behavior is
      undefined if there are Sacred subdirectories associated with other
      scripts.
    run_name: If provided, then only analyze results from Sacred directories
      associated with this run name. `run_name` is compared against the
      "experiment.name" key in `run.json`.
    skip_failed_runs: If True, then filter out runs where the status is FAILED.
    csv_output_path: If provided, then save a CSV output file to this path.
    verbose: If True, then print the dataframe.

  Returns:
    A list of dictionaries used to generate the analysis DataFrame.
  """
  sacred_dirs = sacred_util.filter_subdirs(source_dir)
  sacred_dicts = [sacred_util.SacredDicts.load_from_dir(sacred_dir)
                  for sacred_dir in sacred_dirs]

  if run_name is not None:
    sacred_dicts = filter(
      lambda sd: get(sd.run, "experiment.name") == run_name,
      sacred_dicts)

  if skip_failed_runs:
    sacred_dicts = filter(
      lambda sd: get(sd.run, "status") != "FAILED",
      sacred_dicts)

  rows = []
  for sd in sacred_dicts:
    row = OrderedDict()
    rows.append(row)

    # Use get to prevent exceptions when reading in-progress experiments.
    row["status"] = get(sd.run, "status")
    row["use_gail"] = get(sd.config, "init_trainer_kwargs.use_gail")
    row["env_name"] = get(sd.config, "env_name")
    row["n_expert_demos"] = get(sd.config, "n_expert_demos")
    row["run_name"] = get(sd.run, "experiment.name")

    imit_stats = get(sd.run, "result.imit_stats")
    expert_stats = get(sd.run, "result.expert_stats")
    if imit_stats is not None and expert_stats is not None:
      # Assume that `result.imit_stats` and `result.expert_stats` are
      # formatted correctly.
      row["expert_return_summary"] = _make_return_summary(expert_stats)
      row["imit_return_summary"] = _make_return_summary(imit_stats, "monitor_")
      row["imit_vs_expert_return"] = (expert_stats["return_mean"] /
                                      imit_stats["monitor_return_mean"])
      row["imit_return_mean"] = imit_stats["monitor_return_mean"]
      row["imit_return_std_dev"] = imit_stats["monitor_return_std"]

  df = pd.DataFrame(rows)
  df.to_csv(csv_output_path)

  if verbose:
    print(df.to_string())
  return rows


def _make_return_summary(stats: dict, prefix="") -> str:
  return "{:3g} ± {:3g} (n={})".format(
    stats[f"{prefix}return_mean"],
    stats[f"{prefix}return_std"],
    stats["n_traj"])


def main_console():
  observer = FileStorageObserver.create(
    osp.join('output', 'sacred', 'analyze'))
  analysis_ex.observers.append(observer)
  analysis_ex.run_commandline()


if __name__ == "__main__":
  main_console()
