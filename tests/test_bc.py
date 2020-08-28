"""Tests for Behavioural Cloning (BC)."""

import os

import pytest
import torch as th
from torch.utils import data as th_data

from imitation.algorithms import bc
from imitation.data import rollout, types
from imitation.util import util

ROLLOUT_PATH = "tests/data/expert_models/cartpole_0/rollouts/final.pkl"


@pytest.fixture
def venv():
    env_name = "CartPole-v1"
    venv = util.make_vec_env(env_name, 2)
    return venv


@pytest.fixture(params=[32, 50])
def batch_size(request):
    return request.param


@pytest.fixture
def trainer(batch_size, venv):
    rollouts = types.load(ROLLOUT_PATH)
    dataset = rollout.flatten_trajectories(rollouts)
    dataloader = th_data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return bc.BC(
        venv.observation_space, venv.action_space, expert_dataloader=dataloader
    )


def test_weight_decay_init_error(venv):
    with pytest.raises(ValueError, match=".*weight_decay.*"):
        bc.BC(
            venv.observation_space,
            venv.action_space,
            expert_dataloader=None,
            optimizer_kwargs=dict(weight_decay=1e-4),
        )


def test_bc(trainer: bc.BC, venv):
    sample_until = rollout.min_episodes(15)
    novice_ret_mean = rollout.mean_return(trainer.policy, venv, sample_until)
    trainer.train(n_epochs=1, on_epoch_end=lambda _: print("epoch end"))
    trained_ret_mean = rollout.mean_return(trainer.policy, venv, sample_until)
    # Typically <80 score is bad, >350 is okay. We want an improvement of at
    # least 50 points, which seems like it's not noise.
    assert trained_ret_mean - novice_ret_mean > 50


def test_save_reload(trainer, tmpdir):
    pol_path = os.path.join(tmpdir, "policy.pt")
    var_values = list(trainer.policy.parameters())
    trainer.save_policy(pol_path)
    new_policy = bc.reconstruct_policy(pol_path)
    new_values = list(new_policy.parameters())
    assert len(var_values) == len(new_values)
    for old, new in zip(var_values, new_values):
        assert th.allclose(old, new)
