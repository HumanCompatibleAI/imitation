"""Smoke tests for CLI programs in imitation.scripts.*"""

import os.path as osp
import tempfile

from imitation.scripts.data_collect import data_collect_ex
from imitation.scripts.policy_eval import policy_eval_ex
from imitation.scripts.train import train_ex


def test_data_collect_main():
  """Smoke test for imitation.scripts.data_collect.rollouts_and_policy"""
  run = data_collect_ex.run(
      named_configs=['cartpole', 'fast'],
  )
  assert run.status == 'COMPLETED'


def test_data_collect_rollouts_from_policy():
  """Smoke test for imitation.scripts.data_collect.rollouts_from_policy"""
  with tempfile.TemporaryDirectory(prefix='imitation-data_collect-policy',
                                   ) as tmpdir:
    run = data_collect_ex.run(
        command_name="rollouts_from_policy",
        named_configs=['cartpole', 'fast'],
        config_updates=dict(
          rollout_save_dir=tmpdir,
          policy_path="expert_models/PPO2_CartPole-v1_0",
        ))
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
          'rollout_glob': "tests/data/rollouts/CartPole*.pkl",
      },
      'log_root': 'output/tests/train',
  }
  run = train_ex.run(
      named_configs=['cartpole', 'gail', 'fast'],
      config_updates=config_updates,
  )
  assert run.status == 'COMPLETED'


def test_transfer_learning():
  """Transfer learning smoke test.

  Save a dummy AIRL test reward, then load it for transfer learning."""

  with tempfile.TemporaryDirectory(prefix='imitation-transfer',
                                   ) as tmpdir:
    log_dir_train = osp.join(tmpdir, "train")
    run = train_ex.run(
        named_configs=['cartpole', 'airl', 'fast'],
        config_updates = {
          'init_trainer_kwargs': {
            'rollout_glob': "tests/data/rollouts/CartPole*.pkl",
          },
          'log_root': log_dir_train,
        },
    )
    assert run.status == 'COMPLETED'

    log_dir_data = osp.join(tmpdir, "data_collect")
    discrim_path = osp.join(tmpdir, "checkpoint", "discrim", "final")
    run = data_collect_ex.run(
        named_configs=['cartpole', 'fast'],
        config_updates = {
          'log_root': log_dir_data,
          'discrim_net_airl_path': discrim_path,
        }
    assert run.status == 'COMPLETED'
