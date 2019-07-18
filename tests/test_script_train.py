import tensorflow as tf

from imitation.scripts.train import train_ex


def test_train():
  train_ex.run(
    named_configs=['cartpole', 'gail', 'fast'],
    config_updates={'init_trainer_kwargs':
                    {'rollouts_glob': "tests/data/rollouts/CartPole*.npz"}})
  tf.reset_default_graph()
