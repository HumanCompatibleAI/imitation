"""Smoke tests for CLI programs in imitation.scripts.*"""

from imitation.scripts.data_collect import data_collect_ex
from imitation.scripts.train import train_ex


def test_data_collect():
  """Smoke test for imitation.scripts.data_collect"""
  data_collect_ex.run(
      named_configs=['cartpole', 'fast'],
      # codecov does not like parallel
      config_updates={'parallel': False},
  )


def test_train():
  """Smoke test for imitation.scripts.train"""
  config_updates = {
      'init_trainer_kwargs': {
          # codecov does not like parallel
          'parallel': False,
          'rollouts_glob': "tests/data/rollouts/CartPole*.pkl",
      },
  }
  train_ex.run(
    named_configs=['cartpole', 'gail', 'fast'],
    config_updates=config_updates,
  )
