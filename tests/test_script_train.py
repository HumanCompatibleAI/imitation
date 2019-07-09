import gin
import gin.tf
import tensorflow as tf

from imitation.scripts.train import train_and_plot

gin.parse_config_file("configs/classical_control.gin")
gin.bind_parameter('train_and_plot.env', 'CartPole-v1')
gin.bind_parameter('init_trainer.use_gail', False)


def test_train_and_plot_no_crash():
  train_and_plot(n_epochs=2,
                 n_plots_each_per_epoch=1,
                 n_disc_steps_per_epoch=1,
                 n_gen_steps_per_epoch=1,
                 interactive=False)
  tf.reset_default_graph()


# TODO(shwang): edit notebooks to match new params
