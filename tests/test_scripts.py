"""Smoke tests for CLI programs in imitation.scripts.*

Every test in this file should use `parallel=False` to turn off multiprocessing because
codecov might interact poorly with multiprocessing. The 'fast' named_config for each
experiment implicitly sets parallel=False.
"""

import collections
import pathlib
import sys
import tempfile
from collections import Counter
from typing import List, Optional
from unittest import mock

import pandas as pd
import pytest
import ray.tune as tune
import sacred

from imitation.scripts import (
    analyze,
    eval_policy,
    expert_demos,
    parallel,
    train_adversarial,
    train_bc,
)

ALL_SCRIPTS_MODS = [
    analyze,
    eval_policy,
    expert_demos,
    parallel,
    train_adversarial,
    train_bc,
]

CARTPOLE_TEST_DATA_PATH = pathlib.Path("tests/data/expert_models/cartpole_0/")
CARTPOLE_TEST_ROLLOUT_PATH = CARTPOLE_TEST_DATA_PATH / "rollouts/final.pkl"
CARTPOLE_TEST_POLICY_PATH = CARTPOLE_TEST_DATA_PATH / "policies/final"


@pytest.fixture(autouse=True)
def sacred_capture_use_sys():
    """Set Sacred capture mode to "sys" because default "fd" option leads to error.

    See https://github.com/IDSIA/sacred/issues/289."""
    # TODO(shwang): Stop using non-default "sys" mode once the issue is fixed.
    temp = sacred.SETTINGS["CAPTURE_MODE"]
    sacred.SETTINGS.CAPTURE_MODE = "sys"
    yield
    sacred.SETTINGS.CAPTURE_MODE = temp


@pytest.mark.parametrize("script_mod", ALL_SCRIPTS_MODS)
def test_main_console(script_mod):
    """Smoke tests of main entry point for some cheap coverage."""
    argv = ["sacred-pytest-stub", "print_config"]
    with mock.patch.object(sys, "argv", argv):
        script_mod.main_console()


def test_train_bc_main(tmpdir):
    run = train_bc.train_bc_ex.run(
        named_configs=["fast", "cartpole"],
        config_updates=dict(
            log_root=tmpdir,
            expert_data_src=CARTPOLE_TEST_ROLLOUT_PATH,
        ),
    )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


def test_expert_demos_main(tmpdir):
    """Smoke test for imitation.scripts.expert_demos.rollouts_and_policy."""
    run = expert_demos.expert_demos_ex.run(
        named_configs=["cartpole", "fast"],
        config_updates=dict(
            log_root=tmpdir,
        ),
    )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


def test_expert_demos_rollouts_from_policy(tmpdir):
    """Smoke test for imitation.scripts.expert_demos.rollouts_from_policy."""
    run = expert_demos.expert_demos_ex.run(
        command_name="rollouts_from_policy",
        named_configs=["cartpole", "fast"],
        config_updates=dict(
            log_root=tmpdir,
            rollout_save_path=str(pathlib.Path(tmpdir, "rollouts", "test.pkl")),
            policy_path=CARTPOLE_TEST_POLICY_PATH,
        ),
    )
    assert run.status == "COMPLETED"


EVAL_POLICY_CONFIGS = [
    {"videos": True},
    {"videos": True, "video_kwargs": {"single_video": False}},
    {"reward_type": "zero", "reward_path": "foobar"},
]


@pytest.mark.parametrize("config", EVAL_POLICY_CONFIGS)
def test_eval_policy(config, tmpdir):
    """Smoke test for imitation.scripts.eval_policy."""
    config_updates = {
        "log_root": tmpdir,
    }
    config_updates.update(config)
    run = eval_policy.eval_policy_ex.run(
        config_updates=config_updates, named_configs=["fast"]
    )
    assert run.status == "COMPLETED"
    wrapped_reward = "reward_type" in config
    _check_rollout_stats(run.result, wrapped_reward)


def _check_rollout_stats(stats: dict, wrapped_reward: bool = True):
    """Common assertions for rollout_stats."""
    assert isinstance(stats, dict)
    assert "return_mean" in stats
    assert "monitor_return_mean" in stats
    if wrapped_reward:
        # If the reward is wrapped, then we expect the monitor return
        # to differ.
        assert stats.get("return_mean") != stats.get("monitor_return_mean")
    else:
        assert stats.get("return_mean") == stats.get("monitor_return_mean")


def _check_train_ex_result(result: dict):
    expert_stats = result.get("expert_stats")
    assert isinstance(expert_stats, dict)
    assert "return_mean" in expert_stats
    assert "monitor_return_mean" not in expert_stats

    imit_stats = result.get("imit_stats")
    _check_rollout_stats(imit_stats)


def test_train_adversarial(tmpdir):
    """Smoke test for imitation.scripts.train_adversarial."""
    named_configs = ["cartpole", "gail", "fast"]
    config_updates = {
        "log_root": tmpdir,
        "rollout_path": CARTPOLE_TEST_ROLLOUT_PATH,
        "init_tensorboard": True,
    }
    run = train_adversarial.train_adversarial_ex.run(
        named_configs=named_configs,
        config_updates=config_updates,
    )
    assert run.status == "COMPLETED"
    _check_train_ex_result(run.result)


def test_train_adversarial_algorithm_value_error(tmpdir):
    """Error on bad algorithm arguments."""
    base_named_configs = ["cartpole", "fast"]
    base_config_updates = collections.ChainMap(
        {
            "log_root": tmpdir,
            "rollout_path": CARTPOLE_TEST_ROLLOUT_PATH,
        }
    )

    with pytest.raises(ValueError, match=".*BAD_VALUE.*"):
        train_adversarial.train_adversarial_ex.run(
            named_configs=base_named_configs,
            config_updates=base_config_updates.new_child(dict(algorithm="BAD_VALUE")),
        )

    with pytest.raises(ValueError, match=".*BAD_VALUE.*"):
        train_adversarial.train_adversarial_ex.run(
            named_configs=base_named_configs,
            config_updates=base_config_updates.new_child(
                dict(discrim_net_kwargs={"BAD_VALUE": "bar"})
            ),
        )

    with pytest.raises(ValueError, match=".*BAD_VALUE.*"):
        train_adversarial.train_adversarial_ex.run(
            named_configs=base_named_configs,
            config_updates=base_config_updates.new_child(
                dict(algorithm_kwargs={"BAD_VALUE": "bar"})
            ),
        )

    with pytest.raises(ValueError, match=".*BAD_VALUE.*"):
        train_adversarial.train_adversarial_ex.run(
            named_configs=base_named_configs,
            config_updates=base_config_updates.new_child(
                dict(rollout_path="path/BAD_VALUE")
            ),
        )

    n_traj = 1234567
    with pytest.raises(ValueError, match=f".*{n_traj}.*"):
        train_adversarial.train_adversarial_ex.run(
            named_configs=base_named_configs,
            config_updates=base_config_updates.new_child(dict(n_expert_demos=n_traj)),
        )


def test_transfer_learning(tmpdir):
    """Transfer learning smoke test.

    Saves a dummy AIRL test reward, then loads it for transfer learning.
    """
    tmpdir = pathlib.Path(tmpdir)
    log_dir_train = tmpdir / "train"
    run = train_adversarial.train_adversarial_ex.run(
        named_configs=["cartpole", "airl", "fast"],
        config_updates=dict(
            rollout_path=CARTPOLE_TEST_ROLLOUT_PATH,
            log_dir=log_dir_train,
        ),
    )
    assert run.status == "COMPLETED"
    _check_train_ex_result(run.result)

    _check_rollout_stats(run.result["imit_stats"])

    log_dir_data = tmpdir / "expert_demos"
    discrim_path = log_dir_train / "checkpoints" / "final" / "discrim.pt"
    run = expert_demos.expert_demos_ex.run(
        named_configs=["cartpole", "fast"],
        config_updates=dict(
            log_dir=log_dir_data,
            reward_type="DiscrimNet",
            reward_path=discrim_path,
        ),
    )
    assert run.status == "COMPLETED"
    _check_rollout_stats(run.result)


PARALLEL_CONFIG_UPDATES = [
    dict(
        sacred_ex_name="expert_demos",
        base_named_configs=["cartpole", "fast"],
        n_seeds=2,
        search_space={
            "config_updates": {
                "init_rl_kwargs": {"learning_rate": tune.grid_search([3e-4, 1e-4])},
            },
            "meta_info": {"asdf": "I exist for coverage purposes"},
        },
    ),
    dict(
        sacred_ex_name="train_adversarial",
        base_named_configs=["cartpole", "gail", "fast"],
        base_config_updates={
            # Need absolute path because raylet runs in different working directory.
            "rollout_path": CARTPOLE_TEST_ROLLOUT_PATH.absolute(),
        },
        search_space={
            "config_updates": {
                "algorithm": tune.grid_search(["gail", "airl"]),
                # FIXME(sam): this method of searching for hidden sizes won't
                # work now that I've changed the API for reward/discriminator
                # networks. I think we need a nicer API for building such
                # networks, analogous to the one Stable Baselines has for
                # policies. We should add back architecture search support once
                # we have that new API.
                # "algorithm_kwargs": {
                #     "airl": {
                #         "reward_net_kwargs": {
                #             "build_mlp_kwargs": {
                #                 "hid_sizes": tune.grid_search([[16, 16], [7, 9]]),
                #             },
                #         }
                #     }
                # },
            },
        },
    ),
]

PARALLEL_CONFIG_LOW_RESOURCE = {
    # CI server only has 2 cores.
    "init_kwargs": {"num_cpus": 2},
    # Memory is low enough we only want to run one job at a time.
    "resources_per_trial": {"cpu": 2},
}


@pytest.mark.parametrize("config_updates", PARALLEL_CONFIG_UPDATES)
def test_parallel(config_updates):
    """Hyperparam tuning smoke test."""
    # CI server only has 2 cores
    config_updates = dict(config_updates)
    config_updates.update(PARALLEL_CONFIG_LOW_RESOURCE)
    # No need for TemporaryDirectory because the hyperparameter tuning script
    # itself generates no artifacts, and "debug_log_root" sets inner experiment's
    # log_root="/tmp/parallel_debug/".
    run = parallel.parallel_ex.run(
        named_configs=["debug_log_root"], config_updates=config_updates
    )
    assert run.status == "COMPLETED"


def test_parallel_arg_errors(tmpdir):
    """Error on bad algorithm arguments."""
    base_named_configs = ["debug_log_root"]
    base_config_updates = collections.ChainMap(PARALLEL_CONFIG_LOW_RESOURCE)

    with pytest.raises(TypeError, match=".*Sequence.*"):
        parallel.parallel_ex.run(
            named_configs=base_named_configs,
            config_updates=base_config_updates.new_child(dict(base_named_configs={})),
        )

    with pytest.raises(TypeError, match=".*Mapping.*"):
        parallel.parallel_ex.run(
            named_configs=base_named_configs,
            config_updates=base_config_updates.new_child(dict(base_config_updates=())),
        )

    with pytest.raises(TypeError, match=".*Sequence.*"):
        parallel.parallel_ex.run(
            named_configs=base_named_configs,
            config_updates=base_config_updates.new_child(
                dict(search_space={"named_configs": {}})
            ),
        )

    with pytest.raises(TypeError, match=".*Mapping.*"):
        parallel.parallel_ex.run(
            named_configs=base_named_configs,
            config_updates=base_config_updates.new_child(
                dict(search_space={"config_updates": ()})
            ),
        )


def _generate_test_rollouts(tmpdir: str, env_named_config: str) -> pathlib.Path:
    tmpdir = pathlib.Path(tmpdir)
    expert_demos.expert_demos_ex.run(
        named_configs=[env_named_config, "fast"],
        config_updates=dict(
            log_dir=tmpdir,
        ),
    )
    rollout_path = tmpdir / "rollouts/final.pkl"
    return rollout_path.absolute()


def test_parallel_train_adversarial_custom_env(tmpdir):
    import gym

    try:
        gym.make("Ant-v3")
    except gym.error.DependencyNotInstalled:  # pragma: no cover
        pytest.skip("mujoco_py not available")
    env_named_config = "custom_ant"
    rollout_path = _generate_test_rollouts(tmpdir, env_named_config)

    config_updates = dict(
        sacred_ex_name="train_adversarial",
        n_seeds=1,
        base_named_configs=[env_named_config, "fast"],
        base_config_updates=dict(
            parallel=True,
            num_vec=2,
            rollout_path=rollout_path,
        ),
    )
    config_updates.update(PARALLEL_CONFIG_LOW_RESOURCE)
    run = parallel.parallel_ex.run(
        named_configs=["debug_log_root"], config_updates=config_updates
    )
    assert run.status == "COMPLETED"


def _run_train_adv_for_test_analyze_imit(run_name, sacred_logs_dir, log_dir):
    run = train_adversarial.train_adversarial_ex.run(
        named_configs=["fast", "cartpole"],
        config_updates=dict(
            rollout_path=CARTPOLE_TEST_ROLLOUT_PATH,
            log_dir=log_dir,
            checkpoint_interval=-1,
        ),
        options={"--name": run_name, "--file_storage": sacred_logs_dir},
    )
    return run


def _run_train_bc_for_test_analyze_imit(run_name, sacred_logs_dir, log_dir):
    run = train_bc.train_bc_ex.run(
        named_configs=["fast", "cartpole"],
        config_updates=dict(
            expert_data_src=CARTPOLE_TEST_ROLLOUT_PATH,
            expert_data_src_format="path",
            log_dir=log_dir,
        ),
        options={"--name": run_name, "--file_storage": sacred_logs_dir},
    )
    return run


@pytest.mark.parametrize("run_names", ([], list("adab")))
@pytest.mark.parametrize(
    "run_sacred_fn",
    (
        _run_train_adv_for_test_analyze_imit,
        _run_train_bc_for_test_analyze_imit,
    ),
)
def test_analyze_imitation(tmpdir: str, run_names: List[str], run_sacred_fn):
    sacred_logs_dir = tmpdir = pathlib.Path(tmpdir)

    # Generate sacred logs (other logs are put in separate tmpdir for deletion).
    for i, run_name in enumerate(run_names):
        with tempfile.TemporaryDirectory(prefix="junk") as log_dir:
            run = run_sacred_fn(run_name, sacred_logs_dir, log_dir)
            assert run.status == "COMPLETED"

    # Check that analyze script finds the correct number of logs.
    def check(run_name: Optional[str], count: int) -> None:
        run = analyze.analysis_ex.run(
            command_name="analyze_imitation",
            config_updates=dict(
                source_dir=sacred_logs_dir,
                env_name="CartPole-v1",
                run_name=run_name,
                csv_output_path=tmpdir / "analysis.csv",
                verbose=True,
            ),
        )
        assert run.status == "COMPLETED"
        df = pd.DataFrame(run.result)
        assert df.shape[0] == count

    for run_name, count in Counter(run_names).items():
        check(run_name, count)

    check(None, len(run_names))  # Check total number of logs.


def test_analyze_gather_tb(tmpdir: str):
    config_updates = dict(local_dir=tmpdir, run_name="test")
    config_updates.update(PARALLEL_CONFIG_LOW_RESOURCE)
    parallel_run = parallel.parallel_ex.run(
        named_configs=["generate_test_data"],
        config_updates=config_updates,
    )
    assert parallel_run.status == "COMPLETED"

    run = analyze.analysis_ex.run(
        command_name="gather_tb_directories",
        config_updates=dict(
            source_dir=tmpdir,
        ),
    )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)
    assert run.result["n_tb_dirs"] == 2
