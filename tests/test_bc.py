"""Tests for Behavioural Cloning (BC)."""

from imitation import bc, util


def test_bc():
  env_id = 'CartPole-v1'
  env = util.make_vec_env(env_id, 2)
  expert_algos = util.load_policy(env)
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
