"""Multiprocessed version of train_adversarial."""
from collections import OrderedDict
import csv
import multiprocessing.dummy
import os.path as osp
from typing import Iterable, List, Optional, Tuple

from sacred.observers import FileStorageObserver
import tensorflow as tf

from imitation.scripts.config.multi_train_adversarial import multi_train_ex
from imitation.scripts.train_adversarial import train_ex
import imitation.util.multi as multi_util


@multi_util.args_single_to_star
def _job(
  row: OrderedDict,
  log_dir: str,
  extra_named_configs: Iterable[str],
  extra_config_updates: dict,
) -> Tuple[OrderedDict, dict]:
  """Start a `train_ex` run.

  Params:
    row: A CSV row encoding config settings for this run. Its fields should
      include 'env_config' (str), n_gen_steps_per_epoch (int),
      'n_expert_demos' (int), and 'phase3_seed' (int).
    log_dir: Logs are written in this directory.
    extra_named_configs: Extra named configs to pass to this sacred run.
    extra_config_updates: Extra config updates to pass to this sacred run.

  Returns:
    `row` (same as argument), and `result` (the return value of the run).
  """
  run = train_ex.run
  stdout_path = osp.join(log_dir, "stdout")
  stderr_path = osp.join(log_dir, "stderr")
  run = multi_util.redirect_to_files(run, stdout_path, stderr_path)

  named_configs = [row['env_config']] + list(extra_named_configs)
  config_updates = {
    "log_dir": log_dir,
    "n_gen_steps_per_epoch": int(row['n_gen_steps_per_epoch']),
    "init_trainer_kwargs.n_expert_demos": int(row['n_expert_demos']),
    "seed": int(row['phase3_seed']),
  }
  config_updates.update(**extra_config_updates)

  result = run(
    named_configs=named_configs, config_updates=config_updates).result
  return row, result


@multi_train_ex.main
def benchmark_adversarial_from_csv(
  log_dir: str,
  n_workers: int,
  csv_config_path: str = "experiments/gail_benchmark_config.csv",
  seeds: Iterable[int] = range(3),
  extra_named_configs: Optional[Iterable[str]] = None,
  extra_config_updates: Optional[dict] = None,
  parallel: bool = True,
) -> None:
  """Start several runs of `scripts.train_adversarial` from a CSV file.

  Results are written as a CSV to `f"{log_dir}/results.csv"`.
  The columns of the results file are the columns of the CSV config file,
  followed by "phase3_seed", and then the keys of the
  `train_adversarial.run(...).result`.

  Params:
    log_dir: Main logging directory.
    n_workers: The number of jobs that are processed simultaneously.
    csv_config_path: Path to CSV configuration. Expected columns are
      "config_name", "n_expert_demos", and "n_gen_steps_per_epoch".
    seeds: Every row in `csv_config_path` is run using each of these seeds.
      The seed associated with each row in the output CSV,
      `f"{log_dir}/results.csv"`, is saved in a column called "phase3_seed".
    extra_named_configs: Extra named configs to apply to every call of
      `train_adversarial.run(...)`.
    extra_config_updates: Extra config updates to apply to every call of
      `train_adversarial.run(...)`.
    parallel: If True, then uses `n_workers` processes. If False, then use
      n_workers threads (mostly useful for debugging purposes).
  """
  if extra_named_configs is None:
    extra_named_configs = []
  if extra_config_updates is None:
    extra_config_updates = {}
  if parallel:
    mp = multiprocessing
  else:
    mp = multiprocessing.dummy

  # Construct job args for `Pool.imap_unnordered`.
  # There is one job for every combination of seed and CSV row.
  job_args = []  # type: List[tuple]
  with open(csv_config_path, newline='') as csv_file:
    for row in csv.DictReader(csv_file):
      for seed in seeds:
        new_row = row.copy()  # type: OrderedDict
        new_row["phase3_seed"] = str(seed)
        job_log_dir = osp.join(log_dir,
                               multi_util.path_from_ordered_dict(new_row))
        job_args.append(
          (new_row, job_log_dir, extra_named_configs, extra_config_updates))

  # Run all jobs and write results to "results.csv".
  csv_output_path = osp.join(log_dir, "results.csv")
  with mp.Pool(n_workers) as pool:
    with open(csv_output_path, 'w', newline='') as csv_file:
      writer = None
      for row, results in pool.imap_unordered(_job, job_args):
        results["phase3_log_dir"] = results["log_dir"]
        del results["log_dir"]

        if writer is None:
          fieldnames = list(row.keys()) + list(results.keys())
          writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
          writer.writeheader()

        writer.writerow(dict(**row, **results))
        tf.logging.info(f"Completed job row={row}.")


def main_console():
  obs_path = osp.join('output', 'sacred', 'multi_train_adversarial')
  job_obs_path = osp.join(obs_path, 'jobs')

  multi_train_ex.observers.append(
    FileStorageObserver.create(obs_path))
  train_ex.observers.append(
    FileStorageObserver.create(job_obs_path))

  multi_train_ex.run_commandline()


if __name__ == "__main__":
  main_console()
