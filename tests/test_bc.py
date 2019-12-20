"""Tests for Behavioural Cloning (BC)."""
import os
import pickle

import numpy as np
import tensorflow as tf

from imitation import util
from imitation.algorithms import bc

ROLLOUT_PATH = "tests/data/expert_models/cartpole_0/rollouts/final.pkl"


def make_trainer():
  env_name = 'CartPole-v1'
  env = util.make_vec_env(env_name, 2)
  with open(ROLLOUT_PATH, "rb") as f:
      rollouts = pickle.load(f)
  rollouts = util.rollout.flatten_trajectories(rollouts)
  return bc.BCTrainer(env, expert_demos=rollouts)


def test_bc():
  bc_trainer = make_trainer()
  novice_stats = bc_trainer.test_policy()
  bc_trainer.train(n_epochs=40)
  good_stats = bc_trainer.test_policy(min_episodes=25)
  # novice is bad
  assert novice_stats["return_mean"] < 80.0
  # bc is okay but isn't perfect (for the purpose of this test)
  assert good_stats["return_mean"] > 350.0


def test_save_reload(tmpdir):
  bc_trainer = make_trainer()
  pol_path = os.path.join(tmpdir, 'policy.pkl')
  # just to change the values a little
  bc_trainer.train(n_epochs=1)
  var_values = bc_trainer.sess.run(bc_trainer.policy_variables)
  bc_trainer.save_policy(pol_path)
  with tf.Session() as sess:
    # just make sure it doesn't die
    with tf.variable_scope('new'):
      bc.BCTrainer.reconstruct_policy(pol_path)
    new_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='new')
    new_values = sess.run(new_vars)
    assert len(var_values) == len(new_values)
    for old, new in zip(var_values, new_values):
      assert np.allclose(old, new)
