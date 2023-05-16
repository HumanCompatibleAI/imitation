"""Smoke tests for CLI programs in `imitation.scripts.*`.

Every test in this file should use `parallel=False` to turn off multiprocessing because
codecov might interact poorly with multiprocessing. The 'fast' named_config for each
experiment implicitly sets parallel=False.
"""

import collections
import filecmp
import os
import pathlib
import pickle
import platform
import shutil
import sys
import tempfile
from collections import Counter
from typing import Any, Dict, Generator, List, Mapping, Optional
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import ray.tune as tune
import sacred
import sacred.utils
import stable_baselines3
import torch as th
from stable_baselines3.common import buffers
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from imitation.data import serialize
from imitation.rewards import reward_nets
from imitation.scripts import (
    analyze,
    convert_trajs,
    eval_policy,
    parallel,
    train_adversarial,
    train_imitation,
    train_preference_comparisons,
    train_rl,
)
from imitation.util import networks, util

ALL_SCRIPTS_MODS = [
    analyze,
    eval_policy,
    parallel,
    train_adversarial,
    train_imitation,
    train_preference_comparisons,
    train_rl,
]

TEST_DATA_PATH = util.parse_path("tests/testdata")

if not TEST_DATA_PATH.exists():  # pragma: no cover
    raise RuntimeError(
        "Folder with test data has not been found. Make sure you are "
        "running tests relative to the base imitation project folder.",
    )

CARTPOLE_TEST_DATA_PATH = TEST_DATA_PATH / "expert_models/cartpole_0/"
CARTPOLE_TEST_ROLLOUT_PATH = CARTPOLE_TEST_DATA_PATH / "rollouts/final.npz"
CARTPOLE_TEST_POLICY_PATH = CARTPOLE_TEST_DATA_PATH / "policies/final"

PENDULUM_TEST_DATA_PATH = TEST_DATA_PATH / "expert_models/pendulum_0/"
PENDULUM_TEST_ROLLOUT_PATH = PENDULUM_TEST_DATA_PATH / "rollouts/final.npz"

PICKLE_FMT_ROLLOUT_TEST_DATA_PATH = TEST_DATA_PATH / "pickle_format_rollout.pkl"
NPZ_FMT_ROLLOUT_TEST_DATA_PATH = TEST_DATA_PATH / "npz_format_rollout.npz"


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


ALGO_FAST_CONFIGS = {
    "adversarial": [
        "environment.fast",
        "demonstrations.fast",
        "rl.fast",
        "policy_evaluation.fast",
        "fast",
    ],
    "eval_policy": ["environment.fast", "fast"],
    "imitation": [
        "environment.fast",
        "demonstrations.fast",
        "policy_evaluation.fast",
        "fast",
    ],
    "preference_comparison": [
        "environment.fast",
        "rl.fast",
        "policy_evaluation.fast",
        "fast",
    ],
    "rl": ["environment.fast", "rl.fast", "fast"],
}

RL_SAC_NAMED_CONFIGS = ["rl.sac", "policy.sac"]


@pytest.fixture(
    params=[
        "plain",
        "with_expert_trajectories",
        "warmstart",
        "with_checkpoints",
    ],
)
def preference_comparison_config(request):
    return dict(
        plain={},
        with_expert_trajectories={"trajectory_path": CARTPOLE_TEST_ROLLOUT_PATH},
        warmstart={
            # We're testing preference saving and disabling sampling here as well;
            # having yet another run just for those would be wasteful since they
            # don't interact with warm starting an agent.
            "save_preferences": True,
            "gatherer_kwargs": {"sample": False},
            "agent_path": CARTPOLE_TEST_POLICY_PATH,
            # FIXME(yawen): the policy we load was trained on 8 parallel environments
            #  and for some reason using it breaks if we use just 1 (like would be the
            #  default with the fast named_config)
            "environment": dict(num_vec=8),
        },
        with_checkpoints={
            # Test that we can save checkpoints
            "checkpoint_interval": 1,
        },
    )[request.param]


def test_train_preference_comparisons_main(tmpdir, preference_comparison_config):
    config_updates = dict(logging=dict(log_root=tmpdir))
    sacred.utils.recursive_update(config_updates, preference_comparison_config)
    run = train_preference_comparisons.train_preference_comparisons_ex.run(
        # Note: we have to use the cartpole and not the seals_cartpole config because
        #  the seals_cartpole config needs rollouts with a fixed horizon,
        #  and the saved trajectory rollouts are variable horizon.
        named_configs=["cartpole"] + ALGO_FAST_CONFIGS["preference_comparison"],
        config_updates=config_updates,
    )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


@pytest.mark.parametrize(
    "env_name",
    ["seals_cartpole", "mountain_car", "seals_mountain_car"],
)
def test_train_preference_comparisons_envs_no_crash(tmpdir, env_name):
    """Test envs specified in imitation.scripts.config.train_preference_comparisons."""
    config_updates = dict(logging=dict(log_root=tmpdir))
    run = train_preference_comparisons.train_preference_comparisons_ex.run(
        named_configs=[env_name] + ALGO_FAST_CONFIGS["preference_comparison"],
        config_updates=config_updates,
    )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


def test_train_preference_comparisons_sac(tmpdir):
    config_updates = dict(logging=dict(log_root=tmpdir))
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

    # Make sure rl.sac named_config is called after rl.fast to overwrite
    # rl_kwargs.batch_size to None
    with pytest.raises(Exception, match=".*set 'batch_size' at top-level.*"):
        train_preference_comparisons.train_preference_comparisons_ex.run(
            named_configs=["pendulum"]
            + RL_SAC_NAMED_CONFIGS
            + ALGO_FAST_CONFIGS["preference_comparison"],
            config_updates=config_updates,
        )


def test_train_preference_comparisons_sac_reward_relabel(tmpdir):
    def _run_reward_relabel_sac_preference_comparisons(buffer_cls):
        config_updates = dict(
            logging=dict(log_root=tmpdir),
            rl=dict(
                rl_kwargs=dict(
                    replay_buffer_class=buffer_cls,
                    replay_buffer_kwargs=dict(handle_timeout_termination=True),
                ),
            ),
        )
        run = train_preference_comparisons.train_preference_comparisons_ex.run(
            # make sure rl.sac named_config is called after rl.fast to overwrite
            # rl_kwargs.batch_size to None
            named_configs=["pendulum"]
            + ALGO_FAST_CONFIGS["preference_comparison"]
            + RL_SAC_NAMED_CONFIGS,
            config_updates=config_updates,
        )
        return run

    run = _run_reward_relabel_sac_preference_comparisons(buffers.ReplayBuffer)
    assert run.status == "COMPLETED"
    del run

    with pytest.raises(AssertionError, match=".*only ReplayBuffer is supported.*"):
        _run_reward_relabel_sac_preference_comparisons(buffers.DictReplayBuffer)
    with pytest.raises(AssertionError, match=".*only ReplayBuffer is supported.*"):
        _run_reward_relabel_sac_preference_comparisons(HerReplayBuffer)


@pytest.mark.parametrize(
    "named_configs",
    (
        [],
        ["reward.normalize_output_running"],
        ["reward.normalize_output_disable"],
    ),
)
def test_train_preference_comparisons_reward_named_config(tmpdir, named_configs):
    config_updates = dict(logging=dict(log_root=tmpdir))
    run = train_preference_comparisons.train_preference_comparisons_ex.run(
        named_configs=["seals_cartpole"]
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
            named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["imitation"],
            config_updates=dict(
                logging=dict(log_root=tmpdir),
                demonstrations=dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
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


def test_train_dagger_warmstart(tmpdir):
    run = train_imitation.train_imitation_ex.run(
        command_name="dagger",
        named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["imitation"],
        config_updates=dict(
            logging=dict(log_root=tmpdir),
            demonstrations=dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
        ),
    )
    assert run.status == "COMPLETED"

    log_dir = util.parse_path(run.config["logging"]["log_dir"])
    policy_path = log_dir / "scratch" / "policy-latest.pt"
    run_warmstart = train_imitation.train_imitation_ex.run(
        command_name="dagger",
        named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["imitation"],
        config_updates=dict(
            logging=dict(log_root=tmpdir),
            demonstrations=dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
            bc=dict(agent_path=policy_path),
        ),
    )
    assert run_warmstart.status == "COMPLETED"
    assert isinstance(run_warmstart.result, dict)


def test_train_bc_main_with_none_demonstrations_raises_value_error(tmpdir):
    with pytest.raises(ValueError, match=".*n_expert_demos.*rollout_path.*"):
        train_imitation.train_imitation_ex.run(
            command_name="bc",
            named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["imitation"],
            config_updates=dict(
                logging=dict(log_root=tmpdir),
                demonstrations=dict(n_expert_demos=None),
            ),
        )


def test_train_bc_main_with_demonstrations_from_huggingface(tmpdir):
    train_imitation.train_imitation_ex.run(
        command_name="bc",
        named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["imitation"],
        config_updates=dict(
            logging=dict(log_root=tmpdir),
            demonstrations=dict(rollout_type="ppo-huggingface"),
        ),
    )


def test_train_bc_main_with_demonstrations_raises_error_on_wrong_huggingface_format(
    tmpdir,
):
    with pytest.raises(
        ValueError,
        match="`rollout_type` can either be `local` or of the form .*-huggingface.S*",
    ):
        train_imitation.train_imitation_ex.run(
            command_name="bc",
            named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["imitation"],
            config_updates=dict(
                logging=dict(log_root=tmpdir),
                demonstrations=dict(rollout_type="huggingface-ppo"),
            ),
        )


def test_train_bc_main_with_demonstrations_warns_setting_rollout_type(
    tmpdir,
):
    with pytest.warns(
        RuntimeWarning,
        match="Ignoring `rollout_path` .*",
    ):
        train_imitation.train_imitation_ex.run(
            command_name="bc",
            named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["imitation"],
            config_updates=dict(
                logging=dict(log_root=tmpdir),
                demonstrations=dict(
                    rollout_type="ppo-huggingface",
                    rollout_path="path",
                ),
            ),
        )


@pytest.fixture(
    params=[
        "expert_from_path",
        "expert_from_huggingface",
        "random_expert",
        "zero_expert",
    ],
)
def bc_config(tmpdir, request):
    expert_config = dict(
        expert_from_path=dict(
            policy_type="ppo",
            loader_kwargs=dict(path=CARTPOLE_TEST_POLICY_PATH / "model.zip"),
        ),
        expert_from_huggingface=dict(
            policy_type="ppo-huggingface",
            loader_kwargs=dict(env_id="seals/CartPole-v0"),
        ),
        random_expert=dict(policy_type="random"),
        zero_expert=dict(policy_type="zero"),
    )[request.param]
    return dict(
        command_name="bc",
        named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["imitation"],
        config_updates=dict(
            logging=dict(log_root=tmpdir),
            expert=expert_config,
            demonstrations=dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
        ),
    )


def test_train_bc_main(bc_config):
    run = train_imitation.train_imitation_ex.run(**bc_config)
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


def test_train_bc_warmstart(tmpdir):
    run = train_imitation.train_imitation_ex.run(
        command_name="bc",
        named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["imitation"],
        config_updates=dict(
            logging=dict(log_root=tmpdir),
            demonstrations=dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
            expert=dict(
                policy_type="ppo-huggingface",
                loader_kwargs=dict(env_id="seals/CartPole-v0"),
            ),
        ),
    )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)

    policy_path = util.parse_path(run.config["logging"]["log_dir"]) / "final.th"
    run_warmstart = train_imitation.train_imitation_ex.run(
        command_name="bc",
        named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["imitation"],
        config_updates=dict(
            logging=dict(log_root=tmpdir),
            demonstrations=dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
            bc=dict(agent_path=policy_path),
        ),
    )

    assert run_warmstart.status == "COMPLETED"
    assert isinstance(run_warmstart.result, dict)


@pytest.fixture(params=["cold_start", "warm_start"])
def rl_train_ppo_config(request, tmpdir):
    config = dict(logging=dict(log_root=tmpdir))
    if request.param == "warm_start":
        # FIXME(yawen): the policy we load was trained on 8 parallel environments
        #  and for some reason using it breaks if we use just 1 (like would be the
        #  default with the fast named_config)
        config["agent_path"] = CARTPOLE_TEST_POLICY_PATH / "model.zip"
        config["environment"] = dict(num_vec=8)
    return config


def test_train_rl_main(tmpdir, rl_train_ppo_config):
    """Smoke test for imitation.scripts.train_rl."""
    config_updates = dict(logging=dict(log_root=tmpdir))
    sacred.utils.recursive_update(config_updates, rl_train_ppo_config)
    run = train_rl.train_rl_ex.run(
        named_configs=["cartpole"] + ALGO_FAST_CONFIGS["rl"],
        config_updates=config_updates,
    )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


def test_train_rl_wb_logging(tmpdir):
    """Smoke test for imitation.scripts.ingredients.logging.wandb_logging."""
    with pytest.raises(Exception, match=".*api_key not configured.*"):
        train_rl.train_rl_ex.run(
            named_configs=["cartpole"]
            + ALGO_FAST_CONFIGS["rl"]
            + ["logging.wandb_logging"],
            config_updates=dict(
                logging=dict(log_root=tmpdir),
            ),
        )


def test_train_rl_sac(tmpdir):
    run = train_rl.train_rl_ex.run(
        # make sure rl.sac named_config is called after rl.fast to overwrite
        # rl_kwargs.batch_size to None
        named_configs=["pendulum"] + ALGO_FAST_CONFIGS["rl"] + RL_SAC_NAMED_CONFIGS,
        config_updates=dict(
            logging=dict(log_root=tmpdir),
        ),
    )
    assert run.config["rl"]["rl_cls"] is stable_baselines3.SAC
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


# check if platform is macos

EVAL_POLICY_CONFIGS: List[Dict] = [
    {"reward_type": "zero", "reward_path": "foobar"},
    {"explore_kwargs": {"switch_prob": 1.0, "random_prob": 0.1}},
    {"rollout_save_path": "{log_dir}/rollouts.npz"},
]

if platform.system() != "Darwin":
    EVAL_POLICY_CONFIGS.extend(
        [
            {"videos": True},
            {"videos": True, "video_kwargs": {"single_video": False}},
        ],
    )


@pytest.mark.parametrize("config", EVAL_POLICY_CONFIGS)
def test_eval_policy(config, tmpdir):
    """Smoke test for imitation.scripts.eval_policy."""
    config_updates = dict(logging=dict(log_root=tmpdir))
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
    assert isinstance(imit_stats, dict)
    _check_rollout_stats(imit_stats)


@pytest.mark.parametrize(
    "named_configs",
    (
        [],
        ["policy.normalize_running", "reward.normalize_input_running"],
        ["reward.normalize_input_disable"],
    ),
)
@pytest.mark.parametrize("command", ("airl", "gail"))
def test_train_adversarial(tmpdir, named_configs, command):
    """Smoke test for imitation.scripts.train_adversarial."""
    named_configs = (
        named_configs + ["seals_cartpole"] + ALGO_FAST_CONFIGS["adversarial"]
    )
    config_updates = {
        "logging": dict(log_root=tmpdir),
        "demonstrations": dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
        # TensorBoard logs to get extra coverage
        "algorithm_kwargs": dict(init_tensorboard=True),
    }
    run = train_adversarial.train_adversarial_ex.run(
        command_name=command,
        named_configs=named_configs,
        config_updates=config_updates,
    )
    assert run.status == "COMPLETED"
    _check_train_ex_result(run.result)


@pytest.mark.parametrize("command", ("airl", "gail"))
def test_train_adversarial_warmstart(tmpdir, command):
    named_configs = ["cartpole"] + ALGO_FAST_CONFIGS["adversarial"]
    config_updates = {
        "logging": dict(log_root=tmpdir),
        "demonstrations": dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
    }
    run = train_adversarial.train_adversarial_ex.run(
        command_name=command,
        named_configs=named_configs,
        config_updates=config_updates,
    )

    log_dir = util.parse_path(run.config["logging"]["log_dir"])
    policy_path = log_dir / "checkpoints" / "final" / "gen_policy"

    run_warmstart = train_adversarial.train_adversarial_ex.run(
        command_name=command,
        named_configs=named_configs,
        config_updates={
            "agent_path": policy_path,
            **config_updates,
        },
    )

    assert run_warmstart.status == "COMPLETED"
    _check_train_ex_result(run_warmstart.result)


@pytest.mark.parametrize("command", ("airl", "gail"))
def test_train_adversarial_sac(tmpdir, command):
    """Smoke test for imitation.scripts.train_adversarial."""
    # Make sure rl.sac named_config is called after rl.fast to overwrite
    # rl_kwargs.batch_size to None
    named_configs = (
        ["pendulum"] + ALGO_FAST_CONFIGS["adversarial"] + RL_SAC_NAMED_CONFIGS
    )
    config_updates = {
        "logging": dict(log_root=tmpdir),
        "demonstrations": dict(rollout_path=PENDULUM_TEST_ROLLOUT_PATH),
    }
    run = train_adversarial.train_adversarial_ex.run(
        command_name=command,
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
            "logging": dict(log_root=tmpdir),
            "demonstrations": dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
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
    tmpdir_path = util.parse_path(tmpdir)
    log_dir_train = tmpdir_path / "train"
    run = train_adversarial.train_adversarial_ex.run(
        command_name="airl",
        named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["adversarial"],
        config_updates=dict(
            logging=dict(log_dir=log_dir_train),
            demonstrations=dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
        ),
    )
    assert run.status == "COMPLETED"
    _check_train_ex_result(run.result)

    _check_rollout_stats(run.result["imit_stats"])

    log_dir_data = tmpdir_path / "train_rl"
    reward_path = log_dir_train / "checkpoints" / "final" / "reward_test.pt"
    run = train_rl.train_rl_ex.run(
        named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["rl"],
        config_updates=dict(
            logging=dict(log_dir=log_dir_data),
            reward_type="RewardNet_unshaped",
            reward_path=reward_path,
        ),
    )
    assert run.status == "COMPLETED"
    _check_rollout_stats(run.result)


@pytest.mark.parametrize(
    "named_configs_dict",
    (
        dict(pc=[], rl=[]),
        dict(pc=["rl.sac", "policy.sac"], rl=["rl.sac", "policy.sac"]),
        dict(pc=["reward.reward_ensemble"], rl=[]),
    ),
)
def test_preference_comparisons_transfer_learning(
    tmpdir: str,
    named_configs_dict: Mapping[str, List[str]],
) -> None:
    """Transfer learning smoke test.

    Saves a preference comparisons ensemble reward, then loads it for transfer learning.

    Args:
        tmpdir: Temporary directory to save results to.
        named_configs_dict: Named configs for preference_comparisons and rl.
    """
    tmpdir_path = util.parse_path(tmpdir)

    log_dir_train = tmpdir_path / "train"
    run = train_preference_comparisons.train_preference_comparisons_ex.run(
        named_configs=["pendulum"]
        + ALGO_FAST_CONFIGS["preference_comparison"]
        + named_configs_dict["pc"],
        config_updates=dict(logging=dict(log_dir=log_dir_train)),
    )
    assert run.status == "COMPLETED"

    if "reward.reward_ensemble" in named_configs_dict["pc"]:
        assert run.config["reward"]["net_cls"] is reward_nets.RewardEnsemble
        assert run.config["reward"]["add_std_alpha"] == 0.0
        reward_type = "RewardNet_std_added"
        load_reward_kwargs = {"alpha": -1}
    else:
        reward_type = "RewardNet_unnormalized"
        load_reward_kwargs = {}

    log_dir_data = tmpdir_path / "train_rl"
    reward_path = log_dir_train / "checkpoints" / "final" / "reward_net.pt"
    agent_path = log_dir_train / "checkpoints" / "final" / "policy"
    run = train_rl.train_rl_ex.run(
        named_configs=["pendulum"] + ALGO_FAST_CONFIGS["rl"] + named_configs_dict["rl"],
        config_updates=dict(
            logging=dict(log_dir=log_dir_data),
            reward_type=reward_type,
            reward_path=reward_path,
            load_reward_kwargs=load_reward_kwargs,
            agent_path=agent_path,
        ),
    )
    assert run.status == "COMPLETED"
    _check_rollout_stats(run.result)


def test_train_rl_double_normalization(tmpdir: str, rng):
    venv = util.make_vec_env(
        "CartPole-v1",
        n_envs=1,
        parallel=False,
        rng=rng,
    )
    basic_reward_net = reward_nets.BasicRewardNet(
        venv.observation_space,
        venv.action_space,
    )
    net = reward_nets.NormalizedRewardNet(basic_reward_net, networks.RunningNorm)
    tmppath = os.path.join(tmpdir, "reward.pt")
    th.save(net, tmppath)

    log_dir_data = os.path.join(tmpdir, "train_rl")
    with pytest.warns(
        RuntimeWarning,
        match=r"Applying normalization to already normalized reward function.*",
    ):
        train_rl.train_rl_ex.run(
            named_configs=["cartpole"] + ALGO_FAST_CONFIGS["rl"],
            config_updates=dict(
                logging=dict(log_dir=log_dir_data),
                reward_type="RewardNet_normalized",
                normalize_reward=True,
                reward_path=tmppath,
            ),
        )


def test_train_rl_cnn_policy(tmpdir: str, rng):
    venv = util.make_vec_env(
        "AsteroidsNoFrameskip-v4",
        n_envs=1,
        parallel=False,
        rng=rng,
    )
    net = reward_nets.CnnRewardNet(venv.observation_space, venv.action_space)
    tmppath = os.path.join(tmpdir, "reward.pt")
    th.save(net, tmppath)

    log_dir_data = os.path.join(tmpdir, "train_rl")
    run = train_rl.train_rl_ex.run(
        named_configs=["policy.cnn_policy"] + ALGO_FAST_CONFIGS["rl"],
        config_updates=dict(
            environment=dict(gym_id="AsteroidsNoFrameskip-v4"),
            logging=dict(log_dir=log_dir_data),
            reward_path=tmppath,
        ),
    )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


PARALLEL_CONFIG_UPDATES = [
    dict(
        sacred_ex_name="train_rl",
        base_named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["rl"],
        repeat=2,
        search_space={
            "config_updates": {
                "rl": {"rl_kwargs": {"learning_rate": tune.choice([3e-4, 1e-4])}},
            },
            "meta_info": {"asdf": "I exist for coverage purposes"},
        },
    ),
    dict(
        sacred_ex_name="train_adversarial",
        base_named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["adversarial"],
        base_config_updates={
            # Need absolute path because raylet runs in different working directory.
            "demonstrations.rollout_path": CARTPOLE_TEST_ROLLOUT_PATH.absolute(),
        },
        search_space={
            "command_name": "airl",
            "config_updates": {"total_timesteps": tune.choice([5, 10])},
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
    config_updates.setdefault("base_config_updates", {})["logging.log_root"] = tmpdir
    run = parallel.parallel_ex.run(config_updates=config_updates)
    assert run.status == "COMPLETED"


def test_parallel_arg_errors(tmpdir):
    """Error on bad algorithm arguments."""
    config_updates = dict(PARALLEL_CONFIG_LOW_RESOURCE)
    config_updates.setdefault("base_config_updates", {})["logging.log_root"] = tmpdir
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
    tmpdir_path = util.parse_path(tmpdir)
    train_rl.train_rl_ex.run(
        named_configs=[env_named_config] + ALGO_FAST_CONFIGS["rl"],
        config_updates=dict(
            logging=dict(log_dir=tmpdir),
        ),
    )
    rollout_path = tmpdir_path / "rollouts/final.npz"
    return rollout_path.absolute()


def test_parallel_train_adversarial_custom_env(tmpdir):
    if os.name == "nt":  # pragma: no cover
        pytest.skip(
            "`ray.init()` times out when this test runs concurrently with other "
            "test_parallel tests on Windows (e.g., `pytest -n auto -k test_parallel`)",
        )

    env_named_config = "pendulum"
    rollout_path = _generate_test_rollouts(tmpdir, env_named_config)

    config_updates = dict(
        sacred_ex_name="train_adversarial",
        repeat=2,
        base_named_configs=[env_named_config] + ALGO_FAST_CONFIGS["adversarial"],
        base_config_updates=dict(
            logging=dict(log_root=tmpdir),
            demonstrations=dict(rollout_path=rollout_path),
        ),
        search_space=dict(command_name=tune.choice(["gail"])),
    )
    config_updates.update(PARALLEL_CONFIG_LOW_RESOURCE)
    run = parallel.parallel_ex.run(config_updates=config_updates)
    assert run.status == "COMPLETED"


def _run_train_adv_for_test_analyze_imit(run_name, sacred_logs_dir, log_dir):
    run = train_adversarial.train_adversarial_ex.run(
        command_name="gail",
        named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["adversarial"],
        config_updates=dict(
            logging=dict(log_root=log_dir),
            demonstrations=dict(rollout_path=CARTPOLE_TEST_ROLLOUT_PATH),
            checkpoint_interval=-1,
        ),
        options={"--name": run_name, "--file_storage": sacred_logs_dir},
    )
    return run


def _run_train_bc_for_test_analyze_imit(run_name, sacred_logs_dir, log_dir):
    run = train_imitation.train_imitation_ex.run(
        command_name="bc",
        named_configs=["seals_cartpole"] + ALGO_FAST_CONFIGS["imitation"],
        config_updates=dict(
            logging=dict(log_dir=log_dir),
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
    sacred_logs_dir = tmpdir_path = util.parse_path(tmpdir)

    # Generate sacred logs (other logs are put in separate tmpdir for deletion).
    for run_name in run_names:
        with tempfile.TemporaryDirectory(prefix="junk") as log_dir:
            run = run_sacred_fn(run_name, sacred_logs_dir, log_dir)
            assert run.status == "COMPLETED"

    # Check that analyze script finds the correct number of logs.
    def check(run_name: Optional[str], count: int) -> None:
        run = analyze.analysis_ex.run(
            command_name="analyze_imitation",
            config_updates=dict(
                source_dirs=[sacred_logs_dir],
                env_name="seals/CartPole-v0",
                run_name=run_name,
                csv_output_path=tmpdir_path / "analysis.csv",
                tex_output_path=tmpdir_path / "analysis.tex",
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
    if os.name == "nt":  # pragma: no cover
        pytest.skip("gather_tb uses symlinks: not supported by Windows")
    num_runs = 2
    config_updates: Dict[str, Any] = dict(local_dir=tmpdir, run_name="test")
    config_updates.update(PARALLEL_CONFIG_LOW_RESOURCE)
    config_updates.update(num_samples=num_runs)
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
    assert run.result["n_tb_dirs"] == num_runs


def test_pickle_fmt_rollout_test_data_is_pickle():
    # WHEN
    with open(PICKLE_FMT_ROLLOUT_TEST_DATA_PATH, "rb") as f:
        pickle.load(f)

    # THEN
    # No exception raised


def test_npz_fmt_rollout_test_data_is_npz():
    # WHEN
    np.load(NPZ_FMT_ROLLOUT_TEST_DATA_PATH, allow_pickle=False)

    # THEN
    # No exception raised


@pytest.fixture(
    params=[PICKLE_FMT_ROLLOUT_TEST_DATA_PATH, NPZ_FMT_ROLLOUT_TEST_DATA_PATH],
    ids=["pickle", "npz"],
)
def old_rollouts_file_in_tmp(tmpdir, request) -> Generator[str, None, None]:
    old_rollouts_file = request.param
    shutil.copy(old_rollouts_file, tmpdir)
    yield os.path.join(tmpdir, os.path.basename(old_rollouts_file))


@pytest.fixture()
def new_rollouts_file_in_tmp(tmpdir) -> Generator[pathlib.Path, None, None]:
    # Note: here we just convert some old rollouts to the new format, so we don't need
    #  to keep trajectories in the new format in the testdata folder
    npz_rollouts_in_tmp = shutil.copy(NPZ_FMT_ROLLOUT_TEST_DATA_PATH, tmpdir)
    yield convert_trajs.update_traj_file_in_place(npz_rollouts_in_tmp)


def test_converted_trajectories_equal_original(old_rollouts_file_in_tmp: str):
    # GIVEN
    old_trajs = serialize.load(old_rollouts_file_in_tmp)

    # WHEN
    converted_path = convert_trajs.update_traj_file_in_place(old_rollouts_file_in_tmp)

    # THEN
    converted_trajs = serialize.load(converted_path)

    assert len(old_trajs) == len(converted_trajs)
    for t_old, t_converted in zip(old_trajs, converted_trajs):
        assert t_old == t_converted


def test_convert_trajs_from_current_format_is_idempotent(
    new_rollouts_file_in_tmp: pathlib.Path,
):
    # GIVEN
    original_path = new_rollouts_file_in_tmp.with_suffix(".orig")

    # WHEN
    shutil.copytree(new_rollouts_file_in_tmp, original_path)
    converted_path = convert_trajs.update_traj_file_in_place(new_rollouts_file_in_tmp)

    # THEN
    assert (
        filecmp.dircmp(converted_path, original_path).diff_files == []
    ), "convert_trajs not idempotent"
