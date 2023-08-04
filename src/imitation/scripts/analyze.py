"""Commands to analyze experimental results."""

import itertools
import json
import logging
import os
import pathlib
import tempfile
import warnings
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Set

import pandas as pd
from sacred.observers import FileStorageObserver

import imitation.util.sacred as sacred_util
from imitation.scripts.config.analyze import analysis_ex
from imitation.util import util
from imitation.util.sacred import dict_get_nested as get


@analysis_ex.capture
def _gather_sacred_dicts(
    source_dirs: Sequence[str],
    run_name: Optional[str],
    env_name: Optional[str],
    skip_failed_runs: bool,
) -> List[sacred_util.SacredDicts]:
    """Helper function for parsing and selecting Sacred experiment JSON files.

    Args:
        source_dirs: A directory containing Sacred FileObserver subdirectories
            associated with the `train_adversarial` Sacred script. Behavior is
            undefined if there are Sacred subdirectories associated with other
            scripts. (Captured argument)
        run_name: If provided, then only analyze results from Sacred directories
            associated with this run name. `run_name` is compared against the
            "experiment.name" key in `run.json`. (Captured argument)
        env_name: If provided, then only analyze results from Sacred directories
            associated with this Gym environment ID.
        skip_failed_runs: If True, then filter out runs where the status is FAILED.
            (Captured argument)

    Returns:
        A list of `SacredDicts` corresponding to the selected Sacred directories.
    """
    # e.g. chain.from_iterable([["pathone", "pathtwo"], [], ["paththree"]]) =>
    # ("pathone", "pathtwo", "paththree")
    sacred_dirs = itertools.chain.from_iterable(
        sacred_util.filter_subdirs(util.parse_path(source_dir))
        for source_dir in source_dirs
    )
    sacred_dicts_list = []

    for sacred_dir in sacred_dirs:
        try:
            sacred_dicts_list.append(sacred_util.SacredDicts.load_from_dir(sacred_dir))
        except json.JSONDecodeError:
            warnings.warn(f"Invalid JSON file in {sacred_dir}", RuntimeWarning)

    sacred_dicts: Iterable = sacred_dicts_list
    if run_name is not None:
        sacred_dicts = filter(
            lambda sd: get(sd.run, "experiment.name") == run_name,
            sacred_dicts,
        )

    if env_name is not None:
        sacred_dicts = filter(
            lambda sd: get(sd.config, "environment.gym_id") == env_name,
            sacred_dicts,
        )

    if skip_failed_runs:
        sacred_dicts = filter(
            lambda sd: get(sd.run, "status") != "FAILED",
            sacred_dicts,
        )

    return list(sacred_dicts)


@analysis_ex.command
def gather_tb_directories() -> dict:
    """Gather Tensorboard directories from a `parallel_ex` run.

    The directories are copied to a unique directory in `/tmp/analysis_tb/` under
    subdirectories matching the Tensorboard events' Ray Tune trial names.

    This function calls the helper `_gather_sacred_dicts`, which captures its arguments
    automatically via Sacred. Provide those arguments to select which Sacred
    results to parse.

    Returns:
        A dict with two keys. "gather_dir" (str) is a path to a /tmp/ directory
        containing all the TensorBoard runs filtered from `source_dir`.
        "n_tb_dirs" (int) is the number of TensorBoard directories that were filtered.

    Raises:
        OSError: If the symlink cannot be created.
    """
    tb_analysis_dir = pathlib.Path("/tmp/analysis_tb")
    tb_analysis_dir.mkdir(exist_ok=True)
    tmp_dir = pathlib.Path(tempfile.mkdtemp(dir=tb_analysis_dir))

    tb_dirs_count = 0
    for sd in _gather_sacred_dicts():
        # Expecting a path like "~/ray_results/{run_name}/sacred/1".
        # Want to search for all Tensorboard dirs inside
        # "~/ray_results/{run_name}".
        run_dir = sd.sacred_dir.parent.parent
        run_name = run_dir.name

        # log is what we use as subdirectory in new code.
        # rl, tb, sb_tb all appear in old versions.
        for basename in ["log", "rl", "tb", "sb_tb"]:
            tb_src_dirs = tuple(
                sacred_util.filter_subdirs(
                    run_dir,
                    lambda path: path.name == basename,
                ),
            )
            if tb_src_dirs:
                assert len(tb_src_dirs) == 1, "expect at most one TB dir of each type"
                tb_src_dir = tb_src_dirs[0]

                symlinks_dir = tmp_dir / basename
                symlinks_dir.mkdir(exist_ok=True)

                tb_symlink = symlinks_dir / run_name
                try:
                    tb_symlink.symlink_to(tb_src_dir)
                except OSError as e:
                    if os.name == "nt":  # Windows
                        msg = (
                            "Exception occurred while creating symlink. "
                            "Please ensure that Developer mode is enabled."
                        )
                        raise OSError(msg) from e
                    else:
                        raise e
                tb_dirs_count += 1

    logging.info(f"Symlinked {tb_dirs_count} TensorBoard dirs to {tmp_dir}.")
    logging.info(f"Start Tensorboard with `tensorboard --logdir {tmp_dir}`.")
    return {"n_tb_dirs": tb_dirs_count, "gather_dir": tmp_dir}


def _get_exp_command(sd: sacred_util.SacredDicts) -> str:
    return str(sd.run.get("command"))


def _get_algo_name(sd: sacred_util.SacredDicts) -> str:
    exp_command = _get_exp_command(sd)

    if exp_command == "gail":
        return "GAIL"
    elif exp_command == "airl":
        return "AIRL"
    elif exp_command == "train_bc":
        return "BC"
    elif exp_command == "train_dagger":
        return "DAgger"
    else:
        return f"??exp_command={exp_command}"


def _return_summaries(sd: sacred_util.SacredDicts) -> dict:
    imit_stats = get(sd.run, "result.imit_stats")
    expert_stats = get(sd.run, "result.expert_stats")

    expert_return_summary = None
    if expert_stats is not None:
        expert_return_summary = _make_return_summary(expert_stats)

    imit_return_summary = None
    if imit_stats is not None:
        imit_return_summary = _make_return_summary(imit_stats, "monitor_")

    if imit_stats is not None and expert_stats is not None:
        # Assuming here that `result.imit_stats` and `result.expert_stats` are
        # formatted correctly.
        imit_expert_ratio = (
            imit_stats["monitor_return_mean"] / expert_stats["return_mean"]
        )
    else:
        imit_expert_ratio = None

    return dict(
        expert_stats=expert_stats,
        imit_stats=imit_stats,
        expert_return_summary=expert_return_summary,
        imit_return_summary=imit_return_summary,
        imit_expert_ratio=imit_expert_ratio,
    )


sd_to_table_entry_type = Mapping[str, Callable[[sacred_util.SacredDicts], Any]]

# This dict maps column names to functions that get table entries, given the
# row's unique SacredDicts object.
table_entry_fns: sd_to_table_entry_type = {
    "status": lambda sd: get(sd.run, "status"),
    "exp_command": _get_exp_command,
    "algo": _get_algo_name,
    "env_name": lambda sd: get(sd.config, "environment.gym_id"),
    "n_expert_demos": lambda sd: get(sd.config, "demonstrations.n_expert_demos"),
    "run_name": lambda sd: get(sd.run, "experiment.name"),
    "expert_return_summary": lambda sd: _return_summaries(sd)["expert_return_summary"],
    "imit_return_summary": lambda sd: _return_summaries(sd)["imit_return_summary"],
    "imit_expert_ratio": lambda sd: _return_summaries(sd)["imit_expert_ratio"],
}

# If `verbosity` is at least the length of this list, then we use all table_entry_fns
# as columns of table.
# Otherwise, use only the subset at index `verbosity`. The subset of columns is
# still arranged in the same order as in the `table_entry_fns` dict.
table_verbosity_mapping: List[Set[str]] = []

# verbosity 0
table_verbosity_mapping.append(
    {
        "algo",
        "env_name",
        "expert_return_summary",
        "imit_return_summary",
    },
)

# verbosity 1
table_verbosity_mapping.append(table_verbosity_mapping[-1] | {"n_expert_demos"})

# verbosity 2
table_verbosity_mapping.append(
    table_verbosity_mapping[-1]
    | {"status", "imit_expert_ratio", "exp_command", "run_name"},
)


def _get_table_entry_fns_subset(table_verbosity: int) -> sd_to_table_entry_type:
    assert table_verbosity >= 0
    if table_verbosity >= len(table_verbosity_mapping):
        return table_entry_fns
    else:
        keys_subset = table_verbosity_mapping[table_verbosity]
        return {k: v for k, v in table_entry_fns.items() if k in keys_subset}


@analysis_ex.command
def analyze_imitation(
    csv_output_path: Optional[str],
    tex_output_path: Optional[str],
    print_table: bool,
    table_verbosity: int,
) -> pd.DataFrame:
    """Parse Sacred logs and generate a DataFrame for imitation learning results.

    This function calls the helper `_gather_sacred_dicts`, which captures its arguments
    automatically via Sacred. Provide those arguments to select which Sacred
    results to parse.

    Args:
        csv_output_path: If provided, then save a CSV output file to this path.
        tex_output_path: If provided, then save a LaTeX-format table to this path.
        print_table: If True, then print the dataframe to stdout.
        table_verbosity: Increasing levels of verbosity, from 0 to 3, increase the
            number of columns in the table. Level 3 prints all of the columns available.

    Returns:
        The DataFrame generated from the Sacred logs.
    """
    if table_verbosity == 3:
        # Get column names for which we have get value using make_entry_fn
        # These are same across Level 2 & 3. In Level 3, we additionally add remaining
        #  config columns.
        table_entry_fns_subset = _get_table_entry_fns_subset(2)
    else:
        table_entry_fns_subset = _get_table_entry_fns_subset(table_verbosity)

    output_table = pd.DataFrame()
    for sd in _gather_sacred_dicts():
        if table_verbosity == 3:
            # gets all config columns
            row = pd.json_normalize(sd.config)
        else:
            # create an empty dataframe with a single row
            row = pd.DataFrame(index=[0])

        for col_name, make_entry_fn in table_entry_fns_subset.items():
            row[col_name] = make_entry_fn(sd)

        output_table = pd.concat([output_table, row])

    if len(output_table) > 0:
        output_table.sort_values(by=["algo", "env_name"], inplace=True)

    display_options: Mapping[str, Any] = dict(index=False)
    if csv_output_path is not None:
        output_table.to_csv(csv_output_path, **display_options)
        print(f"Wrote CSV file to {csv_output_path}")
    if tex_output_path is not None:
        s: str = output_table.to_latex(**display_options)
        with open(tex_output_path, "w") as f:
            f.write(s)
        print(f"Wrote TeX file to {tex_output_path}")

    if print_table:
        print(output_table.to_string(**display_options))
    return output_table


def _make_return_summary(stats: dict, prefix="") -> str:
    return "{:3g} ± {:3g} (n={})".format(
        stats[f"{prefix}return_mean"],
        stats[f"{prefix}return_std"],
        stats["n_traj"],
    )


def main_console():
    observer_path = pathlib.Path.cwd() / "output" / "sacred" / "analyze"
    observer = FileStorageObserver(observer_path)
    analysis_ex.observers.append(observer)
    analysis_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
