"""Smoke tests for CLI programs in imitation.scripts.*

Every test in this file should use `parallel=False` to turn off multiprocessing
because codecov might interact poorly with multiprocessing. The 'fast'
named_config for each experiment implicitly sets parallel=False.
"""

import os.path as osp
import tempfile

import pytest

from imitation.scripts.benchmark_adversarial import benchmark_adversarial_ex
from imitation.scripts.eval_policy import eval_policy_ex
from imitation.scripts.expert_demos import expert_demos_ex
from imitation.scripts.train_adversarial import train_ex


def test_expert_demos_main():
  """Smoke test for imitation.scripts.expert_demos.rollouts_and_policy"""
  with tempfile.TemporaryDirectory(prefix='imitation-data_collect-main',
                                   ) as tmpdir:
      run = expert_demos_ex.run(
          named_configs=['cartpole', 'fast'],
          config_updates=dict(
            log_root=tmpdir,
          ),
      )
      assert run.status == 'COMPLETED'


def test_expert_demos_rollouts_from_policy():
  """Smoke test for imitation.scripts.expert_demos.rollouts_from_policy"""
  with tempfile.TemporaryDirectory(prefix='imitation-data_collect-policy',
                                   ) as tmpdir:
    run = expert_demos_ex.run(
        command_name="rollouts_from_policy",
        named_configs=['cartpole', 'fast'],
        config_updates=dict(
          log_root=tmpdir,
          rollout_save_dir=osp.join(tmpdir, "rollouts"),
          policy_path="expert_models/PPO2_CartPole-v1_0",
        ))
  assert run.status == 'COMPLETED'


EVAL_POLICY_CONFIGS = [
    {},
    {'reward_type': 'zero', 'reward_path': 'foobar'},
]


@pytest.mark.parametrize('config', EVAL_POLICY_CONFIGS)
def test_eval_policy(config):
  """Smoke test for imitation.scripts.eval_policy"""
  with tempfile.TemporaryDirectory(prefix='imitation-policy_eval',
                                   ) as tmpdir:
      config_updates = {
          'render': False,
          'log_root': tmpdir,
      }
      config_updates.update(config)
      run = eval_policy_ex.run(config_updates=config_updates,
                               named_configs=['fast'])
      assert run.status == 'COMPLETED'
      assert isinstance(run.result, dict)


def test_train_adversarial():
  """Smoke test for imitation.scripts.train_adversarial"""
  with tempfile.TemporaryDirectory(prefix='imitation-train',
                                   ) as tmpdir:
      config_updates = {
          'init_trainer_kwargs': {
              # Rollouts are small, decrease size of buffer to avoid warning
              'trainer_kwargs': {
                  'n_disc_samples_per_buffer': 50,
              },
          },
          'log_root': tmpdir,
          'rollout_glob': "tests/data/rollouts/CartPole*.pkl",
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
        config_updates=dict(
          rollout_glob="tests/data/rollouts/CartPole*.pkl",
          log_dir=log_dir_train,
        ),
    )
    assert run.status == 'COMPLETED'

    log_dir_data = osp.join(tmpdir, "expert_demos")
    discrim_path = osp.join(log_dir_train, "checkpoints", "final", "discrim")
    run = expert_demos_ex.run(
        named_configs=['cartpole', 'fast'],
        config_updates=dict(
          log_dir=log_dir_data,
          reward_type='DiscrimNetAIRL',
          reward_path=discrim_path,
        ),
    )
    assert run.status == 'COMPLETED'


def test_benchmark_adversarial_from_csv():
  """Smoke test for imitation.scripts.benchmark-adversarial"""
  with tempfile.TemporaryDirectory(prefix='imitation-benchmark-adversarial',
                                   ) as tmpdir:
    run = benchmark_adversarial_ex.run(
      named_configs=['fast'],
      config_updates=dict(
        log_dir=tmpdir,
        csv_config_path="tests/data/adv_benchmark_config.csv",
        extra_config_updates=dict(
          rollout_glob="tests/data/rollouts/CartPole*.pkl",
        ),
      ),
    )
    assert run.status == 'COMPLETED'
