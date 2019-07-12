from imitation.scripts.train import train_ex


def test_train():
  train_ex.run(
    config_updates=dict(
      n_epochs=1, n_epochs_per_plot=1, n_episodes_per_reward_data=1,
      n_disc_steps_per_epoch=1, n_gen_steps_per_epoch=1),
    named_configs=['cartpole', 'gail'])
