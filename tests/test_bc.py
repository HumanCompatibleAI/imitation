"""Tests for Behavioural Cloning (BC)."""

import os

import numpy as np
import pytest
import tensorflow as tf

from imitation.algorithms import bc
from imitation.data import dataset, rollout, types
from imitation.util import util

ROLLOUT_PATH = "tests/data/expert_models/cartpole_0/rollouts/final.pkl"


@pytest.fixture
def venv():
    env_name = "CartPole-v1"
    venv = util.make_vec_env(env_name, 2)
    return venv


@pytest.fixture(params=[False, True])
def trainer(request, session, venv):
    convert_dataset = request.param
    rollouts = types.load(ROLLOUT_PATH)
    data = rollout.flatten_trajectories(rollouts)
    if convert_dataset:
        data_map = {"obs": data.obs, "acts": data.acts}
        data = dataset.RandomDictDataset(data_map)
    return bc.BC(venv.observation_space, venv.action_space, expert_data=data)


def test_bc(trainer: bc.BC, venv):
    sample_until = rollout.min_episodes(25)
    novice_ret_mean = rollout.mean_return(trainer.policy, venv, sample_until)
    trainer.train(n_epochs=40)
    trained_ret_mean = rollout.mean_return(trainer.policy, venv, sample_until)
    # novice is bad
    assert novice_ret_mean < 80.0
    # bc is okay but isn't perfect (for the purpose of this test)
    assert trained_ret_mean > 350.0


def test_save_reload(trainer, tmpdir):
    pol_path = os.path.join(tmpdir, "policy.pkl")
    # just to change the values a little
    trainer.train(n_epochs=1)
    var_values = trainer.sess.run(trainer.policy_variables)
    trainer.save_policy(pol_path)
    with tf.Session() as sess:
        # just make sure it doesn't die
        with tf.variable_scope("new"):
            bc.BC.reconstruct_policy(pol_path)
        new_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="new")
        new_values = sess.run(new_vars)
        assert len(var_values) == len(new_values)
        for old, new in zip(var_values, new_values):
            assert np.allclose(old, new)
