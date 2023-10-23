"""Tests for imitation.util.sacred_file_parsing."""
import json
import pathlib

import imitation.util.sacred_file_parsing as sfp


def _make_sacred_run_dir(
    path: pathlib.Path,
    algo: str,
    env: str,
    status: str = "COMPLETED",
):
    path.mkdir(parents=True, exist_ok=True)
    cfg_file = path / "config.json"
    cfg_file.write_text(json.dumps(dict(environment=dict(gym_id=env))))

    run_file = path / "run.json"
    run_file.write_text(json.dumps(dict(status=status, command=algo)))


def test_load_single_run(tmp_path):
    # GIVEN
    _make_sacred_run_dir(tmp_path / "run1", "ppo", "CartPole-v1")

    # WHEN
    runs = list(sfp.find_sacred_runs(tmp_path))

    # THEN
    assert len(runs) == 1
    assert runs[0][0]["environment"]["gym_id"] == "CartPole-v1"
    assert runs[0][1]["command"] == "ppo"


def test_load_multiple_runs_in_sub_folders(tmp_path):
    # GIVEN
    _make_sacred_run_dir(tmp_path / "run1", "ppo", "CartPole-v1")
    _make_sacred_run_dir(tmp_path / "subfolder1" / "run2", "ppo", "CartPole-v1")
    _make_sacred_run_dir(tmp_path / "subfolder1" / "run3", "ppo", "CartPole-v1")
    _make_sacred_run_dir(tmp_path / "subfolder2" / "run4", "ppo", "CartPole-v1")

    # WHEN
    runs = list(sfp.find_sacred_runs(tmp_path))

    # THEN
    assert len(runs) == 4
    for conf, run in runs:
        assert conf["environment"]["gym_id"] == "CartPole-v1"
        assert run["command"] == "ppo"


def test_loading_only_completed_runs(tmp_path):
    # GIVEN
    _make_sacred_run_dir(tmp_path / "run1", "ppo", "CartPole-v1")
    _make_sacred_run_dir(tmp_path / "run2", "airl", "CartPole-v1", status="FAILED")
    _make_sacred_run_dir(tmp_path / "run3", "ppo", "CartPole-v1", status="COMPLETED")
    _make_sacred_run_dir(tmp_path / "run4", "gail", "CartPole-v1", status="RUNNING")

    # WHEN
    runs = list(sfp.find_sacred_runs(tmp_path, only_completed_runs=True))

    # THEN
    assert len(runs) == 2
    for conf, run in runs:
        assert conf["environment"]["gym_id"] == "CartPole-v1"
        assert run["command"] == "ppo"


def test_grouping_runs_by_algo_and_env(tmp_path):
    # GIVEN
    _make_sacred_run_dir(tmp_path / "run1", "ppo", "CartPole-v1")
    _make_sacred_run_dir(tmp_path / "run2", "airl", "CartPole-v1")
    _make_sacred_run_dir(tmp_path / "run3", "ppo", "CartPole-v1")
    _make_sacred_run_dir(tmp_path / "run4", "gail", "CartPole-v1")
    _make_sacred_run_dir(tmp_path / "run5", "ppo", "LunarLander-v2")
    _make_sacred_run_dir(tmp_path / "run6", "airl", "LunarLander-v2")
    _make_sacred_run_dir(tmp_path / "run7", "ppo", "LunarLander-v2")

    # WHEN
    runs_by_algo_and_env = sfp.group_runs_by_algo_and_env(tmp_path)

    # THEN
    assert set(runs_by_algo_and_env.keys()) == {"ppo", "airl", "gail"}
    assert set(runs_by_algo_and_env["ppo"].keys()) == {"CartPole-v1", "LunarLander-v2"}
    assert set(runs_by_algo_and_env["airl"].keys()) == {"CartPole-v1", "LunarLander-v2"}
    assert set(runs_by_algo_and_env["gail"].keys()) == {"CartPole-v1"}
