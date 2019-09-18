"""Smoke tests for CLI programs in imitation.scripts.*

Every test in this file should use `parallel=False` to turn off multiprocessing
because codecov might interact poorly with multiprocessing. The 'fast'
named_config for each experiment implicitly sets parallel=False.
"""

import os.path as osp
from tempfile import TemporaryDirectory

import pytest
import ray.tune as tune

from imitation.scripts.eval_policy import eval_policy_ex
from imitation.scripts.expert_demos import expert_demos_ex
from imitation.scripts.parallel import parallel_ex
from imitation.scripts.train_adversarial import train_ex


def test_expert_demos_main():
  """Smoke test for imitation.scripts.expert_demos.rollouts_and_policy"""
  with TemporaryDirectory(prefix='imitation-data_collect-main') as tmpdir:
    run = expert_demos_ex.run(
        named_configs=['cartpole', 'fast'],
        config_updates=dict(
          log_root=tmpdir,
        ),
    )
    assert run.status == 'COMPLETED'
    assert isinstance(run.result, dict)


def test_expert_demos_rollouts_from_policy():
  """Smoke test for imitation.scripts.expert_demos.rollouts_from_policy"""
  with TemporaryDirectory(prefix='imitation-data_collect-policy') as tmpdir:
    run = expert_demos_ex.run(
        command_name="rollouts_from_policy",
        named_configs=['cartpole', 'fast'],
        config_updates=dict(
          log_root=tmpdir,
          rollout_save_path=osp.join(tmpdir, "rollouts", "test.pkl"),
          policy_path="tests/data/cartpole_0/policies/final/",
        ))
  assert run.status == 'COMPLETED'


EVAL_POLICY_CONFIGS = [
    {},
    {'reward_type': 'zero', 'reward_path': 'foobar'},
]


@pytest.mark.parametrize('config', EVAL_POLICY_CONFIGS)
def test_eval_policy(config):
  """Smoke test for imitation.scripts.eval_policy"""
  with TemporaryDirectory(prefix='imitation-policy_eval') as tmpdir:
      config_updates = {
          'render': False,
          'log_root': tmpdir,
      }
      config_updates.update(config)
      run = eval_policy_ex.run(config_updates=config_updates,
                               named_configs=['fast'])
      assert run.status == 'COMPLETED'
      wrapped_reward = 'reward_type' in config
      _check_result_rollout_stats(run.result, wrapped_reward)


def _check_result_rollout_stats(result: dict,
                                wrapped_reward: bool = True):
  """Common assertions for results["rollout_stats"]."""
  assert isinstance(result, dict)
  stats = result.get("rollout_stats")
  assert isinstance(stats, dict)
  assert "return_mean" in stats
  assert "monitor_return_mean" in stats
  if wrapped_reward:
    # If the reward is wrapped, then we expect the monitor return
    # to differ.
    assert stats.get("return_mean") != stats.get("monitor_return_mean")
  else:
    assert stats.get("return_mean") == stats.get("monitor_return_mean")


def test_train_adversarial():
  """Smoke test for imitation.scripts.train_adversarial"""
  with TemporaryDirectory(prefix='imitation-train') as tmpdir:
    named_configs = ['cartpole', 'gail', 'fast', 'plots']
    config_updates = {
        'init_trainer_kwargs': {
            # Rollouts are small, decrease size of buffer to avoid warning
            'trainer_kwargs': {
                'n_disc_samples_per_buffer': 50,
            },
        },
        'log_root': tmpdir,
        'rollout_glob': "tests/data/cartpole_0/rollouts/final.pkl",
    }
    run = train_ex.run(
        named_configs=named_configs,
        config_updates=config_updates,
    )
    assert run.status == 'COMPLETED'
    _check_result_rollout_stats(run.result)


def test_transfer_learning():
  """Transfer learning smoke test.

  Save a dummy AIRL test reward, then load it for transfer learning."""

  with TemporaryDirectory(prefix='imitation-transfer') as tmpdir:
    log_dir_train = osp.join(tmpdir, "train")
    run = train_ex.run(
        named_configs=['cartpole', 'airl', 'fast'],
        config_updates=dict(
          rollout_glob="tests/data/cartpole_0/rollouts/final.pkl",
          log_dir=log_dir_train,
        ),
    )
    assert run.status == 'COMPLETED'
    assert isinstance(run.result, dict)

    log_dir_data = osp.join(tmpdir, "expert_demos")
    discrim_path = osp.join(log_dir_train, "checkpoints", "final", "discrim")
    run = expert_demos_ex.run(
        named_configs=['cartpole', 'fast'],
        config_updates=dict(
          log_dir=log_dir_data,
          reward_type='DiscrimNet',
          reward_path=discrim_path,
        ),
    )
    assert run.status == 'COMPLETED'
    _check_result_rollout_stats(run.result)


PARALLEL_CONFIG_UPDATES = [
  dict(
    inner_experiment_name="expert_demos",
    search_space={
      "named_configs": ["cartpole", "fast"],
      "config_updates": {
        "seed": tune.grid_search([0, 1]),
        "init_rl_kwargs": {
          "learning_rate": tune.grid_search([3e-4, 1e-4]),
        },
      }},
  ),
  dict(
    inner_experiment_name="train_adversarial",
    search_space={
      "named_configs": ["cartpole", "gail", "fast"],
      "config_updates": {
        'init_trainer_kwargs': {
          'reward_kwargs': {
            'phi_units': tune.grid_search([[16, 16], [7, 9]]),
          },
        },
        # Need absolute path because raylet runs in different working directory.
        'rollout_glob': osp.abspath("tests/data/cartpole_0/rollouts/final.pkl"),
      }},
  ),
]


@pytest.mark.parametrize("config_updates", PARALLEL_CONFIG_UPDATES)
def test_parallel(config_updates):
  """Hyperparam tuning smoke test."""
  # No need for TemporaryDirectory because the hyperparameter tuning script
  # itself generates no artifacts, and "debug_log_root" sets inner experiment's
  # log_root="/tmp/parallel_debug/".
  run = parallel_ex.run(named_configs=["debug_log_root"],
                        config_updates=config_updates)
  assert run.status == 'COMPLETED'
