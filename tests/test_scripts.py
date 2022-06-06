"""Smoke tests for CLI programs in `imitation.scripts.*`.

Every test in this file should use `parallel=False` to turn off multiprocessing because
codecov might interact poorly with multiprocessing. The 'fast' named_config for each
experiment implicitly sets parallel=False.
"""

import collections
import filecmp
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from typing import List, Optional
from unittest import mock

import pandas as pd
import pytest
import ray.tune as tune
import sacred
import sacred.utils
import stable_baselines3

from imitation.scripts import (
    analyze,
    eval_policy,
    parallel,
    train_adversarial,
    train_imitation,
    train_preference_comparisons,
    train_rl,
)
from imitation.util import networks

ALL_SCRIPTS_MODS = [
    analyze,
    eval_policy,
    parallel,
    train_adversarial,
    train_imitation,
    train_preference_comparisons,
    train_rl,
]

CARTPOLE_TEST_DATA_PATH = pathlib.Path("tests/testdata/expert_models/cartpole_0/")
CARTPOLE_TEST_ROLLOUT_PATH = CARTPOLE_TEST_DATA_PATH / "rollouts/final.pkl"
CARTPOLE_TEST_POLICY_PATH = CARTPOLE_TEST_DATA_PATH / "policies/final"

PENDULUM_TEST_DATA_PATH = pathlib.Path("tests/testdata/expert_models/pendulum_0/")
PENDULUM_TEST_ROLLOUT_PATH = PENDULUM_TEST_DATA_PATH / "rollouts/final.pkl"


@pytest.fixture(autouse=True)
def sacred_capture_use_sys():
    """Set Sacred capture mode to "sys" because default "fd" option leads to error.

    See https://github.com/IDSIA/sacred/issues/289.

    Yields:
        None after setting capture mode; restores it after yield.
    """
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


PREFERENCE_COMPARISON_CONFIGS = [
    {},
    {
        "trajectory_path": CARTPOLE_TEST_ROLLOUT_PATH,
    },
    {
        "agent_path": CARTPOLE_TEST_POLICY_PATH,
        # TODO(ejnnr): the policy we load was trained on 8 parallel environments
        # and for some reason using it breaks if we use just 1 (like would be the
        # default with the fast named_config)
        "common": dict(num_vec=8),
        # We're testing preference saving and disabling sampling here as well;
        # having yet another run just for those would be wasteful since they
        # don't interact with warm starting an agent.
        "save_preferences": True,
        "gatherer_kwargs": {"sample": False},
    },
    {
        "checkpoint_interval": 1,
        # Test that we can save checkpoints
    },
]

ALGO_FAST_CONFIGS = {
    "adversarial": [
        "common.fast",
        "demonstrations.fast",
        "rl.fast",
        "train.fast",
        "fast",
    ],
    "eval_policy": ["common.fast", "fast"],
    "imitation": ["common.fast", "demonstrations.fast", "train.fast", "fast"],
    "preference_comparison": ["common.fast", "rl.fast", "train.fast", "fast"],
    "rl": ["common.fast", "rl.fast", "train.fast", "fast"],
}

RL_SAC_NAMED_CONFIGS = ["rl.sac", "train.sac"]


@pytest.mark.parametrize("config", PREFERENCE_COMPARISON_CONFIGS)
def test_train_preference_comparisons_main(tmpdir, config):
    config_updates = dict(common=dict(log_root=tmpdir))
    sacred.utils.recursive_update(config_updates, config)
    run = train_preference_comparisons.train_preference_comparisons_ex.run(
        named_configs=["cartpole"] + ALGO_FAST_CONFIGS["preference_comparison"],
        config_updates=config_updates,
    )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


def test_train_preference_comparisons_sac(tmpdir):
    config_updates = dict(common=dict(log_root=tmpdir))
    run = train_preference_comparisons.train_preference_comparisons_ex.run(
        # make sure rl.sac named_config is called after rl.fast to overwrite
        # rl_kwargs.batch_size to None
        named_configs=["pendulum"]
        + ALGO_FAST_CONFIGS["preference_comparison"]
        + RL_SAC_NAMED_CONFIGS,
        config_updates=config_updates,
    )
    assert run.config["rl"]["rl_cls"] is stable_baselines3.SAC
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


@pytest.mark.parametrize(
    "named_configs",
    (
        [],
        ["reward.normalize_output_running"],
        ["reward.normalize_output_disable"],
    ),
)
def test_train_preference_comparisons_reward_norm_named_config(tmpdir, named_configs):
    config_updates = dict(common=dict(log_root=tmpdir))
    run = train_preference_comparisons.train_preference_comparisons_ex.run(
        named_configs=["cartpole"]
        + ALGO_FAST_CONFIGS["preference_comparison"]
        + named_configs,
        config_updates=config_updates,
    )
    if "reward.normalize_output_running" in named_configs:
        assert run.config["reward"]["normalize_output_layer"] is networks.RunningNorm
    elif "reward.normalize_output_disable" in named_configs:
        assert run.config["reward"]["normalize_output_layer"] is None
    else:
        assert run.config["reward"]["normalize_output_layer"] is networks.RunningNorm
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


def test_train_dagger_main(tmpdir):
    with pytest.warns(None) as record:
        run = train_imitation.train_imitation_ex.run(
            command_name="dagger",
            named_configs=["cartpole"] + ALGO_FAST_CONFIGS["imitation"],
            config_updates=dict(
                common=dict(log_root=tmpdir),
                demonstrations=dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
                dagger=dict(
                    expert_policy_type="ppo",
                    expert_policy_path=CARTPOLE_TEST_POLICY_PATH,
                ),
            ),
        )
    for warning in record:
        # PyTorch wants writeable arrays.
        # See https://github.com/HumanCompatibleAI/imitation/issues/219
        assert not (
            warning.category == UserWarning
            and "NumPy array is not writeable" in warning.message.args[0]
        )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


def test_train_bc_main(tmpdir):
    run = train_imitation.train_imitation_ex.run(
        command_name="bc",
        named_configs=["cartpole"] + ALGO_FAST_CONFIGS["imitation"],
        config_updates=dict(
            common=dict(log_root=tmpdir),
            demonstrations=dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
        ),
    )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


def test_train_rl_main(tmpdir):
    """Smoke test for imitation.scripts.train_rl.rollouts_and_policy."""
    run = train_rl.train_rl_ex.run(
        named_configs=["cartpole"] + ALGO_FAST_CONFIGS["rl"],
        config_updates=dict(
            common=dict(log_root=tmpdir),
        ),
    )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


def test_train_rl_wb_logging(tmpdir):
    """Smoke test for imitation.scripts.common.common.wandb_logging."""
    with pytest.raises(Exception, match=".*api_key not configured.*"):
        train_rl.train_rl_ex.run(
            named_configs=["cartpole"]
            + ALGO_FAST_CONFIGS["rl"]
            + ["common.wandb_logging"],
            config_updates=dict(
                common=dict(log_root=tmpdir),
            ),
        )


def test_train_rl_sac(tmpdir):
    run = train_rl.train_rl_ex.run(
        # make sure rl.sac named_config is called after rl.fast to overwrite
        # rl_kwargs.batch_size to None
        named_configs=["pendulum"] + ALGO_FAST_CONFIGS["rl"] + RL_SAC_NAMED_CONFIGS,
        config_updates=dict(
            common=dict(log_root=tmpdir),
        ),
    )
    assert run.config["rl"]["rl_cls"] is stable_baselines3.SAC
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


EVAL_POLICY_CONFIGS = [
    {"videos": True},
    {"videos": True, "video_kwargs": {"single_video": False}},
    {"reward_type": "zero", "reward_path": "foobar"},
    {"rollout_save_path": "{log_dir}/rollouts.pkl"},
]


@pytest.mark.parametrize("config", EVAL_POLICY_CONFIGS)
def test_eval_policy(config, tmpdir):
    """Smoke test for imitation.scripts.eval_policy."""
    config_updates = dict(common=dict(log_root=tmpdir))
    config_updates.update(config)
    run = eval_policy.eval_policy_ex.run(
        config_updates=config_updates,
        named_configs=ALGO_FAST_CONFIGS["eval_policy"],
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


@pytest.mark.parametrize(
    "named_configs",
    (
        [],
        ["train.normalize_disable", "reward.normalize_input_disable"],
    ),
)
def test_train_adversarial(tmpdir, named_configs):
    """Smoke test for imitation.scripts.train_adversarial."""
    named_configs = named_configs + ["cartpole"] + ALGO_FAST_CONFIGS["adversarial"]
    config_updates = {
        "common": {
            "log_root": tmpdir,
        },
        "demonstrations": {
            "rollout_path": CARTPOLE_TEST_ROLLOUT_PATH,
        },
        # TensorBoard logs to get extra coverage
        "algorithm_kwargs": {"init_tensorboard": True},
    }
    run = train_adversarial.train_adversarial_ex.run(
        command_name="gail",
        named_configs=named_configs,
        config_updates=config_updates,
    )
    assert run.status == "COMPLETED"
    _check_train_ex_result(run.result)


def test_train_adversarial_sac(tmpdir):
    """Smoke test for imitation.scripts.train_adversarial."""
    # make sure rl.sac named_config is called after rl.fast to overwrite
    # rl_kwargs.batch_size to None
    named_configs = (
        ["pendulum"] + ALGO_FAST_CONFIGS["adversarial"] + RL_SAC_NAMED_CONFIGS
    )
    config_updates = {
        "common": {
            "log_root": tmpdir,
        },
        "demonstrations": {
            "rollout_path": PENDULUM_TEST_ROLLOUT_PATH,
        },
        # TensorBoard logs to get extra coverage
        "algorithm_kwargs": {"init_tensorboard": True},
    }
    run = train_adversarial.train_adversarial_ex.run(
        command_name="gail",
        named_configs=named_configs,
        config_updates=config_updates,
    )
    assert run.config["rl"]["rl_cls"] is stable_baselines3.SAC
    assert run.status == "COMPLETED"
    _check_train_ex_result(run.result)


def test_train_adversarial_algorithm_value_error(tmpdir):
    """Error on bad algorithm arguments."""
    base_named_configs = ["cartpole"] + ALGO_FAST_CONFIGS["adversarial"]
    base_config_updates = collections.ChainMap(
        {
            "common": {
                "log_root": tmpdir,
            },
            "demonstrations": {
                "rollout_path": CARTPOLE_TEST_ROLLOUT_PATH,
            },
        },
    )

    with pytest.raises(TypeError, match=".*BAD_VALUE.*"):
        train_adversarial.train_adversarial_ex.run(
            command_name="gail",
            named_configs=base_named_configs,
            config_updates=base_config_updates.new_child(
                dict(algorithm_kwargs={"BAD_VALUE": "bar"}),
            ),
        )

    with pytest.raises(FileNotFoundError, match=".*BAD_VALUE.*"):
        train_adversarial.train_adversarial_ex.run(
            command_name="gail",
            named_configs=base_named_configs,
            config_updates=base_config_updates.new_child(
                {"demonstrations.rollout_path": "path/BAD_VALUE"},
            ),
        )

    n_traj = 1234567
    with pytest.raises(ValueError, match=f".*{n_traj}.*"):
        train_adversarial.train_adversarial_ex.run(
            command_name="gail",
            named_configs=base_named_configs,
            config_updates=base_config_updates.new_child(
                {"demonstrations.n_expert_demos": n_traj},
            ),
        )


def test_transfer_learning(tmpdir: str) -> None:
    """Transfer learning smoke test.

    Saves a dummy AIRL test reward, then loads it for transfer learning.

    Args:
        tmpdir: Temporary directory to save results to.
    """
    tmpdir = pathlib.Path(tmpdir)
    log_dir_train = tmpdir / "train"
    run = train_adversarial.train_adversarial_ex.run(
        command_name="airl",
        named_configs=["cartpole"] + ALGO_FAST_CONFIGS["adversarial"],
        config_updates=dict(
            common=dict(log_dir=log_dir_train),
            demonstrations=dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
        ),
    )
    assert run.status == "COMPLETED"
    _check_train_ex_result(run.result)

    _check_rollout_stats(run.result["imit_stats"])

    log_dir_data = tmpdir / "train_rl"
    reward_path = log_dir_train / "checkpoints" / "final" / "reward_test.pt"
    run = train_rl.train_rl_ex.run(
        named_configs=["cartpole"] + ALGO_FAST_CONFIGS["rl"],
        config_updates=dict(
            common=dict(log_dir=log_dir_data),
            reward_type="RewardNet_shaped",
            reward_path=reward_path,
        ),
    )
    assert run.status == "COMPLETED"
    _check_rollout_stats(run.result)


PARALLEL_CONFIG_UPDATES = [
    dict(
        sacred_ex_name="train_rl",
        base_named_configs=["cartpole"] + ALGO_FAST_CONFIGS["rl"],
        n_seeds=2,
        search_space={
            "config_updates": {
                "rl": {"rl_kwargs": {"learning_rate": tune.grid_search([3e-4, 1e-4])}},
            },
            "meta_info": {"asdf": "I exist for coverage purposes"},
        },
    ),
    dict(
        sacred_ex_name="train_adversarial",
        base_named_configs=["cartpole"] + ALGO_FAST_CONFIGS["adversarial"],
        base_config_updates={
            # Need absolute path because raylet runs in different working directory.
            "demonstrations.rollout_path": CARTPOLE_TEST_ROLLOUT_PATH.absolute(),
        },
        search_space={
            "command_name": tune.grid_search(["gail", "airl"]),
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
def test_parallel(config_updates, tmpdir):
    """Hyperparam tuning smoke test."""
    # CI server only has 2 cores
    config_updates = dict(config_updates)
    config_updates.update(PARALLEL_CONFIG_LOW_RESOURCE)
    config_updates.setdefault("base_config_updates", {})["common.log_root"] = tmpdir
    run = parallel.parallel_ex.run(config_updates=config_updates)
    assert run.status == "COMPLETED"


def test_parallel_arg_errors(tmpdir):
    """Error on bad algorithm arguments."""
    config_updates = dict(PARALLEL_CONFIG_LOW_RESOURCE)
    config_updates.setdefault("base_config_updates", {})["common.log_root"] = tmpdir
    config_updates = collections.ChainMap(config_updates)

    with pytest.raises(TypeError, match=".*Sequence.*"):
        parallel.parallel_ex.run(
            config_updates=config_updates.new_child(dict(base_named_configs={})),
        )

    with pytest.raises(TypeError, match=".*Mapping.*"):
        parallel.parallel_ex.run(
            config_updates=config_updates.new_child(dict(base_config_updates=())),
        )

    with pytest.raises(TypeError, match=".*Sequence.*"):
        parallel.parallel_ex.run(
            config_updates=config_updates.new_child(
                dict(search_space={"named_configs": {}}),
            ),
        )

    with pytest.raises(TypeError, match=".*Mapping.*"):
        parallel.parallel_ex.run(
            config_updates=config_updates.new_child(
                dict(search_space={"config_updates": ()}),
            ),
        )


def _generate_test_rollouts(tmpdir: str, env_named_config: str) -> pathlib.Path:
    tmpdir = pathlib.Path(tmpdir)
    train_rl.train_rl_ex.run(
        named_configs=[env_named_config] + ALGO_FAST_CONFIGS["rl"],
        config_updates=dict(
            common=dict(log_dir=tmpdir),
        ),
    )
    rollout_path = tmpdir / "rollouts/final.pkl"
    return rollout_path.absolute()


def test_parallel_train_adversarial_custom_env(tmpdir):
    import gym

    try:
        gym.make("seals/Ant-v0")
    except gym.error.DependencyNotInstalled:  # pragma: no cover
        pytest.skip("mujoco_py not available")
    env_named_config = "seals_ant"
    rollout_path = _generate_test_rollouts(tmpdir, env_named_config)

    config_updates = dict(
        sacred_ex_name="train_adversarial",
        n_seeds=1,
        base_named_configs=[env_named_config] + ALGO_FAST_CONFIGS["adversarial"],
        base_config_updates=dict(
            common=dict(log_root=tmpdir),
            demonstrations=dict(rollout_path=rollout_path),
        ),
        search_space=dict(command_name="gail"),
    )
    config_updates.update(PARALLEL_CONFIG_LOW_RESOURCE)
    run = parallel.parallel_ex.run(config_updates=config_updates)
    assert run.status == "COMPLETED"


def _run_train_adv_for_test_analyze_imit(run_name, sacred_logs_dir, log_dir):
    run = train_adversarial.train_adversarial_ex.run(
        command_name="gail",
        named_configs=["cartpole"] + ALGO_FAST_CONFIGS["adversarial"],
        config_updates=dict(
            common=dict(log_root=log_dir),
            demonstrations=dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
            checkpoint_interval=-1,
        ),
        options={"--name": run_name, "--file_storage": sacred_logs_dir},
    )
    return run


def _run_train_bc_for_test_analyze_imit(run_name, sacred_logs_dir, log_dir):
    run = train_imitation.train_imitation_ex.run(
        command_name="bc",
        named_configs=["cartpole"] + ALGO_FAST_CONFIGS["imitation"],
        config_updates=dict(
            common=dict(log_dir=log_dir),
            demonstrations=dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
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
                source_dirs=[sacred_logs_dir],
                env_name="CartPole-v1",
                run_name=run_name,
                csv_output_path=tmpdir / "analysis.csv",
                tex_output_path=tmpdir / "analysis.tex",
                print_table=True,
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
            source_dirs=[tmpdir],
        ),
    )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)
    assert run.result["n_tb_dirs"] == 2


def test_convert_trajs_in_place(tmpdir: str):
    shutil.copy(CARTPOLE_TEST_ROLLOUT_PATH, tmpdir)
    tmp_path = os.path.join(tmpdir, os.path.basename(CARTPOLE_TEST_ROLLOUT_PATH))
    exit_code = subprocess.call(
        ["python", "-m", "imitation.scripts.convert_trajs_in_place", tmp_path],
    )
    assert exit_code == 0

    shutil.copy(tmp_path, tmp_path + ".new")
    exit_code = subprocess.call(
        ["python", "-m", "imitation.scripts.convert_trajs_in_place", tmp_path],
    )
    assert exit_code == 0

    assert filecmp.cmp(
        tmp_path,
        tmp_path + ".new",
    ), "convert_trajs_in_place not idempotent"
