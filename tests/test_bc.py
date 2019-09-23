"""Tests for Behavioural Cloning (BC)."""
from imitation import util
from imitation.algorithms import bc


def test_bc():
  env_id = 'CartPole-v1'
  env = util.make_vec_env(env_id, 2)
  rollouts = util.rollout.load_trajectories(
    "tests/data/cartpole_0/rollouts/final.pkl")
  rollouts = util.rollout.flatten_trajectories(rollouts)
  bc_trainer = bc.BCTrainer(env, expert_demos=rollouts)
  novice_stats = bc_trainer.test_policy()
  bc_trainer.train(n_epochs=40)
  good_stats = bc_trainer.test_policy(min_episodes=25)
  # novice is bad
  assert novice_stats["return_mean"] < 80.0
  # bc is okay but isn't perfect (for the purpose of this test)
  assert good_stats["return_mean"] > 350.0
