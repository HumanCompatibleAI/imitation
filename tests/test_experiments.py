"""Smoke tests for bash scripts in experiments/."""

import glob
import os
import subprocess
from typing import List

import pytest

from imitation.data import types

SCRIPT_NAMES = (
    "bc_benchmark.sh",
    "dagger_benchmark.sh",
    "benchmark_and_table.sh",
    "imit_benchmark.sh",
    "rollouts_from_policies.sh",
    "transfer_learn_benchmark.sh",
)

BENCHMARKING_DIR = types.parse_path("benchmarking")

if not BENCHMARKING_DIR.exists():  # pragma: no cover
    raise RuntimeError(
        "The benchmarking/ folder has not been found. Make sure you are "
        "running tests relative to the base imitation project folder.",
    )

EXPERIMENTS_DIR = types.parse_path("experiments")
COMMANDS_PY_PATH = EXPERIMENTS_DIR / "commands.py"


def _get_benchmarking_path(benchmarking_file):
    return os.path.join(BENCHMARKING_DIR.stem, benchmarking_file)


def _run_commands_from_flags(**kwargs) -> List[str]:
    """Run commands.py with flags derived from the given `kwargs`.

    This is a helper function to reduce boilerplate code in the tests
    for commands.py.

    Each key-value pair in kwargs corresponds to one flag.
    If the value in the key-value is True, then the flag has the form "--key".
    Otherwise, the flag has the form "--key=value".

    For example, _run_commands_from_flags(name="baz", remote=True)
    will execute the following python command:

    python experiments/commands.py --name=baz --remote

    The function will return the parsed output of executing that command.

    Args:
        kwargs: keyword arguments from which to derive flags.

    Returns:
        A list where the ith entry of the list is the ith line printed out
        by the python command.
    """
    flags = []

    # Add some defaults so that most tests can skip specifying these flags.
    if "name" not in kwargs:
        flags.append("--name=run0")

    if "cfg_pattern" not in kwargs:
        cfg_pattern = _get_benchmarking_path("fast_dagger_seals_cartpole.json")
        flags.append("--cfg_pattern=" + cfg_pattern)

    if "output_dir" not in kwargs:
        flags.append("--output_dir=output")

    # Build the flags.
    for key in kwargs:
        flag = "--" + key
        if (not isinstance(kwargs[key], bool)) and kwargs[key]:
            flag += "=" + kwargs[key]
        flags.append(flag)

    py_command = f"python {COMMANDS_PY_PATH} " + " ".join(flags)

    completed_process = subprocess.run(
        py_command,
        shell=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        check=True,
    )

    # Each line of the formatted stdout is a command.
    formatted_stdout = completed_process.stdout.decode("ascii").strip()
    commands = formatted_stdout.split("\n")

    return commands


def test_commands_local_config():
    commands = _run_commands_from_flags()
    assert len(commands) == 1
    expected = (
        "python -m imitation.scripts.train_imitation dagger "
        "--capture=sys --name=run0 "
        "--file_storage=output/sacred/"
        "$USER-cmd-run0-dagger-0-8bf911a8 "
        "with benchmarking/fast_dagger_seals_cartpole.json "
        "seed=0 logging.log_root=output"
    )
    assert commands[0] == expected


def test_commands_local_config_runs(tmpdir):
    output_dir = types.parse_path(tmpdir).as_posix()
    commands = _run_commands_from_flags(output_dir=output_dir)
    assert len(commands) == 1
    expected = (
        "python -m imitation.scripts.train_imitation dagger "
        "--capture=sys --name=run0 "
        f"--file_storage={output_dir}/sacred/"
        "$USER-cmd-run0-dagger-0-8bf911a8 "
        "with benchmarking/fast_dagger_seals_cartpole.json "
        f"seed=0 logging.log_root={output_dir}"
    )
    assert commands[0] == expected
    completed_process = subprocess.run(
        commands[0],
        shell=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        check=True,
    )
    assert completed_process.returncode == 0


def test_commands_local_config_with_custom_flags():
    commands = _run_commands_from_flags(
        name="baz",
        seeds="1",
        output_dir="/foo/bar",
    )
    assert len(commands) == 1
    expected = (
        "python -m imitation.scripts.train_imitation dagger "
        "--capture=sys --name=baz "
        "--file_storage=/foo/bar/sacred/"
        "$USER-cmd-baz-dagger-1-8bf911a8 "
        "with benchmarking/fast_dagger_seals_cartpole.json "
        "seed=1 logging.log_root=/foo/bar"
    )
    assert commands[0] == expected


def test_commands_hofvarpnir_config():
    commands = _run_commands_from_flags(output_dir="/data/output", remote=True)
    assert len(commands) == 1
    expected = (
        "ctl job run --name $USER-cmd-run0-dagger-0-c3ac179d "
        "--command "
        "python\\ -m\\ imitation.scripts.train_imitation\\ dagger\\ "
        "--capture=sys\\ --name=run0\\ "
        "--file_storage=/data/output/sacred/"
        "$USER-cmd-run0-dagger-0-c3ac179d\\ "
        "with\\ /data/imitation/benchmarking/fast_dagger_seals_cartpole.json\\ "
        "seed=0\\ logging.log_root=/data/output --container hacobe/devbox:imitation "
        "--login --high-priority --force-pull --never-restart"
    )
    assert commands[0] == expected


def test_commands_hofvarpnir_config_with_custom_flags():
    commands = _run_commands_from_flags(
        name="baz",
        remote_cfg_dir="/bas/bat",
        seeds="1",
        output_dir="/foo/bar",
        container="bam",
        remote=True,
    )
    assert len(commands) == 1
    expected = (
        "ctl job run --name $USER-cmd-baz-dagger-1-345d0f8a "
        "--command "
        "python\\ -m\\ imitation.scripts.train_imitation\\ dagger\\ "
        "--capture=sys\\ --name=baz\\ "
        "--file_storage=/foo/bar/sacred/"
        "$USER-cmd-baz-dagger-1-345d0f8a\\ "
        "with\\ /bas/bat/fast_dagger_seals_cartpole.json\\ "
        "seed=1\\ logging.log_root=/foo/bar --container bam "
        "--login --high-priority --force-pull --never-restart"
    )
    assert commands[0] == expected


def test_commands_bc_config():
    cfg_pattern = _get_benchmarking_path("example_bc_seals_ant_best_hp_eval.json")
    commands = _run_commands_from_flags(cfg_pattern=cfg_pattern)
    assert len(commands) == 1
    expected = (
        "python -m imitation.scripts.train_imitation bc "
        "--capture=sys --name=run0 "
        "--file_storage=output/sacred/"
        "$USER-cmd-run0-bc-0-138a1475 "
        "with benchmarking/example_bc_seals_ant_best_hp_eval.json "
        "seed=0 logging.log_root=output"
    )
    assert commands[0] == expected


def test_commands_dagger_config():
    cfg_pattern = _get_benchmarking_path("example_dagger_seals_ant_best_hp_eval.json")
    commands = _run_commands_from_flags(cfg_pattern=cfg_pattern)
    assert len(commands) == 1
    expected = (
        "python -m imitation.scripts.train_imitation dagger "
        "--capture=sys --name=run0 "
        "--file_storage=output/sacred/"
        "$USER-cmd-run0-dagger-0-6a49161a "
        "with benchmarking/example_dagger_seals_ant_best_hp_eval.json "
        "seed=0 logging.log_root=output"
    )
    assert commands[0] == expected


def test_commands_gail_config():
    cfg_pattern = _get_benchmarking_path("example_gail_seals_ant_best_hp_eval.json")
    commands = _run_commands_from_flags(cfg_pattern=cfg_pattern)
    assert len(commands) == 1
    expected = (
        "python -m imitation.scripts.train_adversarial gail "
        "--capture=sys --name=run0 "
        "--file_storage=output/sacred/"
        "$USER-cmd-run0-gail-0-3ec8154d "
        "with benchmarking/example_gail_seals_ant_best_hp_eval.json "
        "seed=0 logging.log_root=output"
    )
    assert commands[0] == expected


def test_commands_airl_config():
    cfg_pattern = _get_benchmarking_path("example_airl_seals_ant_best_hp_eval.json")
    commands = _run_commands_from_flags(cfg_pattern=cfg_pattern)
    assert len(commands) == 1
    expected = (
        "python -m imitation.scripts.train_adversarial airl "
        "--capture=sys --name=run0 "
        "--file_storage=output/sacred/"
        "$USER-cmd-run0-airl-0-400e1558 "
        "with benchmarking/example_airl_seals_ant_best_hp_eval.json "
        "seed=0 logging.log_root=output"
    )
    assert commands[0] == expected


def test_commands_multiple_configs():
    # Test a more complicated `cfg_pattern`.
    cfg_pattern = _get_benchmarking_path("*.json")
    commands = _run_commands_from_flags(cfg_pattern=cfg_pattern)
    assert len(commands) == len(glob.glob(cfg_pattern))


def test_commands_multiple_configs_multiple_seeds():
    cfg_pattern = _get_benchmarking_path("*.json")
    seeds = "0,1,2"
    commands = _run_commands_from_flags(
        cfg_pattern=cfg_pattern,
        seeds="0,1,2",
    )
    n_configs = len(glob.glob(cfg_pattern))
    n_seeds = len(seeds.split(","))
    assert len(commands) == (n_configs * n_seeds)


@pytest.mark.parametrize(
    "script_name",
    SCRIPT_NAMES,
)
def test_experiments_fast(script_name: str):
    """Quickly check that experiments run successfully on fast mode."""
    if os.name == "nt":  # pragma: no cover
        pytest.skip("bash shell scripts not ported to Windows.")
    exit_code = subprocess.call([f"./{EXPERIMENTS_DIR.stem}/{script_name}", "--fast"])
    assert exit_code == 0
