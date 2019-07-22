"""Smoke tests for CLI programs in imitation.scripts.*"""

from imitation.scripts.data_collect import data_collect_ex
from imitation.scripts.policy_eval import policy_eval_ex
from imitation.scripts.train import train_ex


def test_data_collect():
  """Smoke test for imitation.scripts.data_collect"""
  run = data_collect_ex.run(
      named_configs=['cartpole', 'fast'],
      # codecov does not like parallel
      config_updates={'parallel': False},
  )
  assert run.status == 'COMPLETED'


def test_policy_eval():
  """Smoke test for imitation.scripts.policy_eval"""
  config_updates = {
      'render': False,
      'log_root': 'output/tests/policy_eval',
  }
  run = policy_eval_ex.run(config_updates=config_updates,
                           named_configs=['fast'])
  assert run.status == 'COMPLETED'
  assert isinstance(run.result, dict)


def test_train():
  """Smoke test for imitation.scripts.train"""
  config_updates = {
      'init_trainer_kwargs': {
          # codecov does not like parallel
          'parallel': False,
          'rollout_glob': "tests/data/rollouts/CartPole*.pkl",
      },
      'log_root': 'output/tests/train',
  }
  run = train_ex.run(
      named_configs=['cartpole', 'gail', 'fast'],
      config_updates=config_updates,
  )
  assert run.status == 'COMPLETED'
