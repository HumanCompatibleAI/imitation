import tensorflow as tf

from imitation.scripts.policy_eval import policy_eval_ex
from imitation.scripts.train import train_ex


def test_policy_eval():
  """Smoke test for imitation.scripts.policy_eval"""
  run = policy_eval_ex.run(config_updates={'render': False},
                           named_configs=['fast'])
  assert run.status == 'COMPLETED'
  assert isinstance(run.result, dict)


def test_train():
  train_ex.run(named_configs=['cartpole', 'gail', 'debug'])
  tf.reset_default_graph()
