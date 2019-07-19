"""Tests for Behavioural Cloning (BC)."""
from imitation import bc, util


def test_bc():
  env_id = 'CartPole-v1'
  env = util.make_vec_env(env_id, 2)
  rollouts = util.rollout.load_transitions(
    "tests/data/rollouts/CartPole-v1*.pkl")[:3]
  bc_trainer = bc.BCTrainer(
      env, expert_rollouts=rollouts, n_expert_timesteps=2000)
  novice_stats = bc_trainer.test_policy()
  bc_trainer.train(n_epochs=40)
  good_stats = bc_trainer.test_policy()
  # novice is bad
  assert novice_stats["return_mean"] < 100.0
  # bc is okay but isn't perfect (for the purpose of this test)
  assert good_stats["return_mean"] > 200.0
