import logging
import os
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import List, Optional

import pandas as pd
from sacred.observers import FileStorageObserver

import imitation.util.sacred as sacred_util
from imitation.scripts.config.analyze import analysis_ex
from imitation.util.sacred import dict_get_nested as get


@analysis_ex.command
def gather_tb_directories(
    source_dir: str,
    run_name: Optional[str],
    env_name: Optional[str],
    skip_failed_runs: bool,
) -> dict:
    """Gather Tensorboard directories from a `parallel_ex` run.

    The directories are copied to a unique directory in `/tmp/analysis_tb/` under
    subdirectories matching the Tensorboard events' Ray Tune trial names.

    Undocumented arguments are the same as in `analyze_imitation()`.

    Args:
      source_dir: A local_dir for Ray. For example, `~/ray_results/`.

    Returns:
      A dict with two keys. "gather_dir" (str) is a path to a /tmp/
      directory containing all the TensorBoard runs filtered from `source_dir`.
      "n_tb_dirs" (int) is the number of TensorBoard directories that were
      filtered.
    """
    sacred_dicts = _get_sacred_dicts(source_dir, run_name, env_name, skip_failed_runs)
    os.makedirs("/tmp/analysis_tb", exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir="/tmp/analysis_tb/")

    tb_dirs_count = 0
    for sd in sacred_dicts:
        # Expecting a path like "~/ray_results/{run_name}/sacred/1".
        # Want to search for all Tensorboard dirs inside
        # "~/ray_results/{run_name}".
        sacred_dir = sd.sacred_dir.rstrip("/")
        run_dir = osp.dirname(osp.dirname(sacred_dir))
        run_name = osp.basename(run_dir)

        # "tb" is TensorBoard directory built by our codebase. "sb_tb" is Stable
        # Baselines TensorBoard directory. There should be at most one of each
        # directory.
        for basename in ["rl", "tb", "sb_tb"]:
            tb_src_dirs = tuple(
                sacred_util.filter_subdirs(
                    run_dir, lambda path: osp.basename(path) == basename
                )
            )
            if tb_src_dirs:
                assert len(tb_src_dirs) == 1, "expect at most one TB dir of each type"
                tb_src_dir = tb_src_dirs[0]

                symlinks_dir = osp.join(tmp_dir, basename)
                os.makedirs(symlinks_dir, exist_ok=True)

                tb_symlink = osp.join(symlinks_dir, run_name)
                os.symlink(tb_src_dir, tb_symlink)
                tb_dirs_count += 1

    logging.info(f"Symlinked {tb_dirs_count} TensorBoard dirs to {tmp_dir}.")
    logging.info(f"Start Tensorboard with `tensorboard --logdir {tmp_dir}`.")
    return {"n_tb_dirs": tb_dirs_count, "gather_dir": tmp_dir}


@analysis_ex.command
def analyze_imitation(
    source_dir: str,
    run_name: Optional[str],
    env_name: Optional[str],
    skip_failed_runs: bool,
    csv_output_path: Optional[str],
    verbose: bool,
) -> pd.DataFrame:
    """Parse Sacred logs and generate a DataFrame for imitation learning results.

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
    sacred_dicts = _get_sacred_dicts(source_dir, run_name, env_name, skip_failed_runs)

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
            row["imit_expert_ratio"] = (
                imit_stats["monitor_return_mean"] / expert_stats["return_mean"]
            )

    df = pd.DataFrame(rows)
    if csv_output_path is not None:
        df.to_csv(csv_output_path)
    if verbose:
        print(df.to_string())
    return df


def _make_return_summary(stats: dict, prefix="") -> str:
    return "{:3g} ± {:3g} (n={})".format(
        stats[f"{prefix}return_mean"], stats[f"{prefix}return_std"], stats["n_traj"]
    )


def _get_sacred_dicts(
    source_dir: str, run_name: str, env_name: str, skip_failed_runs: bool
) -> List[sacred_util.SacredDicts]:
    sacred_dirs = sacred_util.filter_subdirs(source_dir)
    sacred_dicts = [
        sacred_util.SacredDicts.load_from_dir(sacred_dir) for sacred_dir in sacred_dirs
    ]

    if run_name is not None:
        sacred_dicts = filter(
            lambda sd: get(sd.run, "experiment.name") == run_name, sacred_dicts
        )

    if env_name is not None:
        sacred_dicts = filter(
            lambda sd: get(sd.config, "env_name") == env_name, sacred_dicts
        )

    if skip_failed_runs:
        sacred_dicts = filter(
            lambda sd: get(sd.run, "status") != "FAILED", sacred_dicts
        )

    return list(sacred_dicts)


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "analyze"))
    analysis_ex.observers.append(observer)
    analysis_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
