"""Tests for Behavioural Cloning (BC)."""

import gin
import tensorflow as tf

# `init_trainer` import required for parsing configs/test.gin.
from imitation import bc, util
from imitation.util.trainer import init_trainer  # noqa: F401

gin.parse_config_file("configs/test.gin")
tf.logging.set_verbosity(tf.logging.INFO)


def test_bc():
  env_id = 'CartPole-v1'
  policy_dir = gin.query_parameter('init_trainer.policy_dir')
  env = util.make_vec_env(env_id, 2)
  expert_algos = util.load_policy(env, basedir=policy_dir)
  if not expert_algos:
    raise ValueError(env)
  bc_trainer = bc.BCTrainer(
      env, expert_trainers=expert_algos, n_expert_timesteps=2000)
  novice_stats = bc_trainer.test_policy()
  bc_trainer.train(n_epochs=40)
  good_stats = bc_trainer.test_policy()
  # novice is bad
  assert novice_stats["return_mean"] < 100.0
  # bc is okay but isn't perfect (for the purpose of this test)
  assert good_stats["return_mean"] > 200.0
