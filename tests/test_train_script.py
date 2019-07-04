import tensorflow as tf

from imitation.scripts.train import train_and_plot


def test_train_and_plot_no_crash():
    train_and_plot(n_epochs=2,
                   n_epochs_per_plot=1,
                   n_disc_steps_per_epoch=1,
                   n_gen_steps_per_epoch=1,
                   n_episodes_per_reward_data=2,
                   n_disc_samples_per_buffer=10,
                   interactive=False)
    tf.reset_default_graph()
