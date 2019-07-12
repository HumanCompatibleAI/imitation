import tensorflow as tf

from imitation.scripts.train import train_ex


def test_train():
  train_ex.run(named_configs=['cartpole', 'gail', 'debug'])
  tf.reset_default_graph()
