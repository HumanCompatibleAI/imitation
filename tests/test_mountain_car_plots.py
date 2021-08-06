"""Smoke tests for imitation.analysis.mountain_car_plots module."""

import pathlib
import pickle

import pytest
from matplotlib import pyplot as plt

from imitation.analysis import mountain_car_plots
from imitation.data import rollout
from imitation.policies import serialize
from imitation.scripts import train_adversarial
from imitation.util import util


@pytest.fixture
def venv():
    return util.make_vec_env("MountainCar-v0")


@pytest.fixture
def rand_policy(venv):
    return util.init_rl(venv)


@pytest.fixture
def trajs(venv, rand_policy):
    return rollout.generate_trajectories(
        rand_policy, venv, sample_until=rollout.min_episodes(5)
    )


def fake_reward_fn(obs, acts, next_obs, steps):
    """Debug reward function.

    If the heatmap code has the correct shape and row-order, then we should expect
    moving up the y-axis to dramatically increase reward and moving right to slightly
    increase reward. You can confirm this effect by calling `plt.show()` inside
    `test_smoke_make_heatmap`.
    """
    pos, vel = obs[:, 0], obs[:, 1]
    return vel * 100 + pos


def test_smoke_make_heatmap(trajs):
    """Smoke test."""
    for act in range(mountain_car_plots.MC_NUM_ACTS):
        fig = mountain_car_plots.make_heatmap(
            act, fake_reward_fn, gen_trajs=trajs, exp_trajs=trajs
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_smoke_plot_reward_vs_time(trajs):
    """Smoke test."""
    trajs_dict = dict(expert=trajs, generator=trajs)
    fig = mountain_car_plots.plot_reward_vs_time(
        trajs_dict, fake_reward_fn, {"expert": "tab:orange"}
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_batch_reward_heatmaps(trajs, tmpdir, rand_policy):
    """Check that `batch_reward_heatmaps` builds a figure for each checkpoint."""
    tmpdir = pathlib.Path(tmpdir)

    # Save dummy mountain car expert and rollouts.
    expert_policy = rand_policy
    expert_policy_path = tmpdir / "expert_policy"
    serialize.save_stable_model(str(expert_policy_path), expert_policy)

    rollout_path = tmpdir / "rollout.pkl"
    with open(rollout_path, "wb") as f:
        pickle.dump(trajs, f)

    # Generate reward function and generator policy checkpoints.
    log_dir = tmpdir / "train_adversarial"
    run = train_adversarial.train_adversarial_ex.run(
        named_configs=["mountain_car", "fast"],
        config_updates=dict(
            rollout_path=rollout_path,
            checkpoint_interval=1,
            log_dir=(tmpdir / "train_adversarial"),
        ),
    )
    assert run.status == "COMPLETED"
    checkpoints_dir = log_dir / "checkpoints"
    assert checkpoints_dir.is_dir()

    # Finally generate batched figures from checkpoints.
    fig_dict = mountain_car_plots.batch_reward_heatmaps(
        checkpoints_dir, exp_trajs=trajs
    )

    n_checkpoints = len(list(checkpoints_dir.iterdir()))
    n_expected_figs = mountain_car_plots.MC_NUM_ACTS * n_checkpoints
    assert len(fig_dict) == n_expected_figs
    for fig in fig_dict.values():
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
