"""Tests for commands.py + Smoke tests for bash scripts in experiments/."""

import glob
import os
import pathlib
import subprocess
from typing import List

import pytest

SCRIPT_NAMES = (
    "bc_benchmark.sh",
    "dagger_benchmark.sh",
    "benchmark_and_table.sh",
    "imit_benchmark.sh",
    "rollouts_from_policies.sh",
    "transfer_learn_benchmark.sh",
)

THIS_DIR = pathlib.Path(__file__).absolute().parent
BENCHMARKING_DIR = THIS_DIR.parent / "benchmarking"
EXPERIMENTS_DIR = THIS_DIR.parent / "experiments"
COMMANDS_PY_PATH = EXPERIMENTS_DIR / "commands.py"

EXPECTED_LOCAL_CONFIG_TEMPLATE = """python -m imitation.scripts.train_imitation dagger \
--capture=sys --name=run0 --file_storage={output_dir}/sacred/\
$USER-cmd-run0-dagger-0-8bf911a8 \
with benchmarking/fast_dagger_seals_cartpole.json \
seed=0 logging.log_root={output_dir}"""

EXPECTED_HOFVARPNIR_CONFIG_TEMPLATE = """ctl job run \
--name $USER-cmd-run0-dagger-0-c3ac179d \
--command "python -m imitation.scripts.train_imitation dagger \
--capture=sys --name=run0 --file_storage={output_dir}/sacred/\
$USER-cmd-run0-dagger-0-c3ac179d \
with /data/imitation/benchmarking/fast_dagger_seals_cartpole.json \
seed=0 logging.log_root={output_dir}" \
--container hacobe/devbox:imitation \
--login --force-pull --never-restart --gpu 0 \
--shared-host-dir-mount /data"""


def _get_benchmarking_path(benchmarking_file):
    return os.path.join(BENCHMARKING_DIR.stem, benchmarking_file)


def _run_commands_from_flags(**kwargs) -> List[str]:
    """Run commands.py with flags derived from the given `kwargs`.

    This is a helper function to reduce boilerplate code in the tests
    for commands.py.

    Each key-value pair in kwargs corresponds to one flag.
    If the value in the key-value is True, then the flag has the form "--key".
    If the value in the key-value is a list, then the flag has the form
    "--key value[0] value[1] ..."
    Otherwise, the flag has the form "--key=value".

    E.g., _run_commands_from_flags(name="baz", seeds=[0, 1], remote=True)
    will execute the following python command:

    python experiments/commands.py --name=baz --seeds 0 1 --remote

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
        if isinstance(kwargs[key], bool) and kwargs[key]:
            flag = f"--{key}"
        elif isinstance(kwargs[key], list):
            value = " ".join(map(str, kwargs[key]))
            flag = f"--{key} {value}"
        else:
            flag = f"--{key}={kwargs[key]}"
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
    if os.name == "nt":  # pragma: no cover
        pytest.skip("commands.py not ported to Windows.")
    commands = _run_commands_from_flags()
    assert len(commands) == 1
    expected = EXPECTED_LOCAL_CONFIG_TEMPLATE.format(output_dir="output")
    assert commands[0] == expected


def test_commands_local_config_runs(tmpdir):
    if os.name == "nt":  # pragma: no cover
        pytest.skip("commands.py not ported to Windows.")
    commands = _run_commands_from_flags(output_dir=tmpdir)
    assert len(commands) == 1
    expected = EXPECTED_LOCAL_CONFIG_TEMPLATE.format(output_dir=tmpdir)
    assert commands[0] == expected
    completed_process = subprocess.run(
        commands[0],
        shell=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        check=True,
    )
    assert completed_process.returncode == 0
    assert (tmpdir / "dagger" / "seals-CartPole-v0").exists()
    assert (tmpdir / "sacred").exists()


def test_commands_local_config_with_custom_flags():
    if os.name == "nt":  # pragma: no cover
        pytest.skip("commands.py not ported to Windows.")
    commands = _run_commands_from_flags(
        name="baz",
        seeds=1,
        output_dir="/foo/bar",
    )
    assert len(commands) == 1
    expected = """python -m imitation.scripts.train_imitation dagger \
--capture=sys --name=baz --file_storage=/foo/bar/sacred/\
$USER-cmd-baz-dagger-1-8bf911a8 \
with benchmarking/fast_dagger_seals_cartpole.json \
seed=1 logging.log_root=/foo/bar"""
    assert commands[0] == expected


def test_commands_hofvarpnir_config():
    if os.name == "nt":  # pragma: no cover
        pytest.skip("commands.py not ported to Windows.")
    output_dir = "/data/output"
    commands = _run_commands_from_flags(output_dir=output_dir, remote=True)
    assert len(commands) == 1
    expected = EXPECTED_HOFVARPNIR_CONFIG_TEMPLATE.format(output_dir=output_dir)
    assert commands[0] == expected


def test_commands_hofvarpnir_config_with_custom_flags():
    if os.name == "nt":  # pragma: no cover
        pytest.skip("commands.py not ported to Windows.")
    commands = _run_commands_from_flags(
        name="baz",
        remote_cfg_dir="/bas/bat",
        seeds=1,
        output_dir="/foo/bar",
        container="bam",
        remote=True,
    )
    assert len(commands) == 1
    expected = """ctl job run --name $USER-cmd-baz-dagger-1-345d0f8a \
--command "python -m imitation.scripts.train_imitation dagger \
--capture=sys --name=baz --file_storage=/foo/bar/sacred/\
$USER-cmd-baz-dagger-1-345d0f8a \
with /bas/bat/fast_dagger_seals_cartpole.json \
seed=1 logging.log_root=/foo/bar" --container bam \
--login --force-pull --never-restart --gpu 0 \
--shared-host-dir-mount /data"""
    assert commands[0] == expected


def test_commands_local_config_with_special_characters_in_flags(tmpdir):
    if os.name == "nt":  # pragma: no cover
        pytest.skip("commands.py not ported to Windows.")
    # Simulate running commands.py with the following flag:
    # --output_dir="\"/tmp/.../foo bar\""
    # And generating a training command with the following flag:
    # --logging.log_root="/tmp/.../foo bar"
    #
    # If we didn't enclose the directory in quotes as in:
    # --output_dir=/tmp/.../foo bar
    # Then we would get an "unrecognized arguments: bar" error
    # trying to run commands.py.
    #
    # Or if we enclosed the directory in double quotes as in:
    # --output_dir="/tmp/.../foo bar"
    # Then we would generate a training command with the following flag:
    # --logging.log_root=/tmp/.../foo bar
    # And we would get an error trying to run the generated command.
    output_subdir = "foo bar"
    unquoted_output_dir = (tmpdir / f"{output_subdir}").strpath
    output_dir = '"\\"' + unquoted_output_dir + '\\""'
    commands = _run_commands_from_flags(output_dir=output_dir)
    assert len(commands) == 1
    # The extra double quotes are removed in the generated command.
    # It has the following flag:
    # --logging.log_root="/tmp/.../foo bar"
    # So it can run without an error introduced by the space.
    expected = EXPECTED_LOCAL_CONFIG_TEMPLATE.format(
        output_dir='"' + unquoted_output_dir + '"',
    )
    assert commands[0] == expected


def test_commands_hofvarpnir_config_with_special_characters_in_flags(tmpdir):
    if os.name == "nt":  # pragma: no cover
        pytest.skip("commands.py not ported to Windows.")
    # See the comments in
    # test_commands_local_config_with_special_characters_in_flags
    # for a discussion of special characters in the flag values.
    output_subdir = "foo bar"
    unquoted_output_dir = (tmpdir / f"{output_subdir}").strpath
    output_dir = '"\\"' + unquoted_output_dir + '\\""'
    commands = _run_commands_from_flags(output_dir=output_dir, remote=True)
    assert len(commands) == 1
    # Make sure double quotes are escaped in the training script command
    # because the training script command is itself enclosed in double
    # quotes within the cluster command.
    expected = EXPECTED_HOFVARPNIR_CONFIG_TEMPLATE.format(
        output_dir='\\"' + unquoted_output_dir + '\\"',
    )
    assert commands[0] == expected


def test_commands_bc_config():
    if os.name == "nt":  # pragma: no cover
        pytest.skip("commands.py not ported to Windows.")
    cfg_pattern = _get_benchmarking_path("bc_seals_ant_best_hp_eval.json")
    commands = _run_commands_from_flags(cfg_pattern=cfg_pattern)
    assert len(commands) == 1
    expected = """python -m imitation.scripts.train_imitation bc \
--capture=sys --name=run0 --file_storage=output/sacred/\
$USER-cmd-run0-bc-0-138a1475 \
with benchmarking/bc_seals_ant_best_hp_eval.json \
seed=0 logging.log_root=output"""
    assert commands[0] == expected


def test_commands_dagger_config():
    if os.name == "nt":  # pragma: no cover
        pytest.skip("commands.py not ported to Windows.")
    cfg_pattern = _get_benchmarking_path("dagger_seals_ant_best_hp_eval.json")
    commands = _run_commands_from_flags(cfg_pattern=cfg_pattern)
    assert len(commands) == 1
    expected = """python -m imitation.scripts.train_imitation dagger \
--capture=sys --name=run0 --file_storage=output/sacred/\
$USER-cmd-run0-dagger-0-6a49161a \
with benchmarking/dagger_seals_ant_best_hp_eval.json \
seed=0 logging.log_root=output"""
    assert commands[0] == expected


def test_commands_gail_config():
    if os.name == "nt":  # pragma: no cover
        pytest.skip("commands.py not ported to Windows.")
    cfg_pattern = _get_benchmarking_path("gail_seals_ant_best_hp_eval.json")
    commands = _run_commands_from_flags(cfg_pattern=cfg_pattern)
    assert len(commands) == 1
    expected = """python -m imitation.scripts.train_adversarial gail \
--capture=sys --name=run0 --file_storage=output/sacred/\
$USER-cmd-run0-gail-0-3ec8154d \
with benchmarking/gail_seals_ant_best_hp_eval.json \
seed=0 logging.log_root=output"""
    assert commands[0] == expected


def test_commands_airl_config():
    if os.name == "nt":  # pragma: no cover
        pytest.skip("commands.py not ported to Windows.")
    cfg_pattern = _get_benchmarking_path("airl_seals_ant_best_hp_eval.json")
    commands = _run_commands_from_flags(cfg_pattern=cfg_pattern)
    assert len(commands) == 1
    expected = """python -m imitation.scripts.train_adversarial airl \
--capture=sys --name=run0 \
--file_storage=output/sacred/$USER-cmd-run0-airl-0-400e1558 \
with benchmarking/airl_seals_ant_best_hp_eval.json \
seed=0 logging.log_root=output"""
    assert commands[0] == expected


def test_commands_multiple_configs():
    if os.name == "nt":  # pragma: no cover
        pytest.skip("commands.py not ported to Windows.")
    # Test a more complicated `cfg_pattern`.
    cfg_pattern = _get_benchmarking_path("*.json")
    commands = _run_commands_from_flags(cfg_pattern=cfg_pattern)
    assert len(commands) == len(glob.glob(cfg_pattern))


def test_commands_multiple_configs_multiple_seeds():
    if os.name == "nt":  # pragma: no cover
        pytest.skip("commands.py not ported to Windows.")
    cfg_pattern = _get_benchmarking_path("*.json")
    seeds = [0, 1, 2]
    commands = _run_commands_from_flags(
        cfg_pattern=cfg_pattern,
        seeds=seeds,
    )
    n_configs = len(glob.glob(cfg_pattern))
    n_seeds = len(seeds)
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
