import tensorflow as tf

from imitation.scripts.train import train_exp

def test_train_and_plot_no_crash():
  train_exp.run(n_epochs=2,
                n_epochs_per_plot=1,
                n_disc_steps_per_epoch=1,
                n_gen_steps_per_epoch=1,
                interactive=False)
  tf.reset_default_graph()
