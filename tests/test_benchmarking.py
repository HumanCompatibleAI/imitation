"""Tests for config files in benchmarking/ folder."""
import glob
import os
import pathlib

import pytest

from imitation.scripts import train_adversarial, train_imitation

THIS_DIR = pathlib.Path(__file__).absolute().parent
BENCHMARKING_DIR = THIS_DIR.parent / "benchmarking"


@pytest.mark.parametrize(
    "command_name",
    ["bc", "dagger", "airl", "gail"],
)
def test_benchmarking_configs(tmpdir, command_name):
    # We test the configs using the print_config command,
    # because running the configs requires MuJoCo.
    # Requiring MuJoCo to run the tests adds too much complexity.
    if command_name in ("bc", "dagger"):
        ex = train_imitation.train_imitation_ex
    elif command_name in ("airl", "gail"):
        ex = train_adversarial.train_adversarial_ex
    cfg_pattern = os.path.join(BENCHMARKING_DIR, f"example_{command_name}_*.json")
    cfg_files = glob.glob(cfg_pattern)
    assert len(cfg_files) == 5, "There should be 1 config file for each of environment."
    for i, cfg_file in enumerate(cfg_files):
        cfg_name = f"{tmpdir.basename}_{i}"
        ex.add_named_config(cfg_name, cfg_file)
        run = ex.run(command_name="print_config", named_configs=[cfg_name])
        assert run.status == "COMPLETED"
