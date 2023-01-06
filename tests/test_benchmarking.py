"""Tests for config files in benchmarking/ folder."""
import pytest

from imitation.data import types
from imitation.scripts import train_adversarial, train_imitation

ALGO_FAST_CONFIGS = {
    "adversarial": [
        "environment.fast",
        "demonstrations.fast",
        "rl.fast",
        "train.fast",
        "fast",
    ],
    "eval_policy": ["environment.fast", "fast"],
    "imitation": ["environment.fast", "demonstrations.fast", "train.fast", "fast"],
    "preference_comparison": ["environment.fast", "rl.fast", "train.fast", "fast"],
    "rl": ["environment.fast", "rl.fast", "train.fast", "fast"],
}

BENCHMARKING_DIR = types.parse_path("benchmarking")

if not BENCHMARKING_DIR.exists():  # pragma: no cover
    raise RuntimeError(
        "The benchmarking/ folder has not been found. Make sure you are "
        "running tests relative to the base imitation project folder.",
    )


@pytest.mark.parametrize("command_name", ["bc", "dagger"])
def test_benchmarking_imitation_config_runs(tmpdir, command_name):
    config_file = "example_" + command_name + "_seals_half_cheetah_best_hp_eval.json"
    config_path = BENCHMARKING_DIR / config_file
    train_imitation.train_imitation_ex.add_named_config(
        tmpdir.basename,
        config_path.as_posix(),
    )
    run = train_imitation.train_imitation_ex.run(
        command_name=command_name,
        named_configs=[tmpdir.basename] + ALGO_FAST_CONFIGS["imitation"],
        config_updates=dict(
            bc_train_kwargs=dict(n_epochs=None),
            logging=dict(log_root=tmpdir),
        ),
    )
    assert run.status == "COMPLETED"


@pytest.mark.parametrize("command_name", ["airl", "gail"])
def test_benchmarking_adversarial_config_runs(tmpdir, command_name):
    config_file = "example_" + command_name + "_seals_half_cheetah_best_hp_eval.json"
    config_path = BENCHMARKING_DIR / config_file
    train_adversarial.train_adversarial_ex.add_named_config(
        tmpdir.basename,
        config_path.as_posix(),
    )
    run = train_adversarial.train_adversarial_ex.run(
        command_name=command_name,
        named_configs=[tmpdir.basename] + ALGO_FAST_CONFIGS["adversarial"],
        config_updates=dict(
            logging=dict(log_root=tmpdir),
        ),
    )
    assert run.status == "COMPLETED"
