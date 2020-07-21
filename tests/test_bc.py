"""Tests for Behavioural Cloning (BC)."""

import os

import pytest
import torch as th

from imitation.algorithms import bc
from imitation.data import datasets, rollout, types
from imitation.util import util

ROLLOUT_PATH = "tests/data/expert_models/cartpole_0/rollouts/final.pkl"


@pytest.fixture
def venv():
    env_name = "CartPole-v1"
    venv = util.make_vec_env(env_name, 2)
    return venv


@pytest.fixture(params=[False, True])
def trainer(request, venv):
    convert_dataset = request.param
    rollouts = types.load(ROLLOUT_PATH)
    data = rollout.flatten_trajectories(rollouts)
    if convert_dataset:
        data = datasets.TransitionsDictDatasetAdaptor(
            data, datasets.EpochOrderDictDataset
        )
    return bc.BC(venv.observation_space, venv.action_space, expert_data=data)


def test_bc(trainer: bc.BC, venv):
    sample_until = rollout.min_episodes(15)
    novice_ret_mean = rollout.mean_return(trainer.policy, venv, sample_until)
    trainer.train(n_epochs=1)
    trained_ret_mean = rollout.mean_return(trainer.policy, venv, sample_until)
    # Typically <80 score is bad, >350 is okay. We want an improvement of at
    # least 50 points, which seems like it's not noise.
    assert trained_ret_mean - novice_ret_mean > 50


def test_train_from_random_dict_dataset(venv):
    # make sure that we can construct BC instance & train from a RandomDictDataset
    rollouts = types.load(ROLLOUT_PATH)
    data = rollout.flatten_trajectories(rollouts)
    data = datasets.TransitionsDictDatasetAdaptor(data, datasets.RandomDictDataset)
    trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=data)
    trainer.train(n_epochs=1)


def test_save_reload(trainer, tmpdir):
    pol_path = os.path.join(tmpdir, "policy.pt")
    var_values = list(trainer.policy.parameters())
    trainer.save_policy(pol_path)
    new_policy = bc.reconstruct_policy(pol_path)
    new_values = list(new_policy.parameters())
    assert len(var_values) == len(new_values)
    for old, new in zip(var_values, new_values):
        assert th.allclose(old, new)
