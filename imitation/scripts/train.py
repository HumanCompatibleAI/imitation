"""Train an IRL algorithm and plot its output.

Can be used as a CLI script, or the `train_and_plot` function can be called
directly.
"""

from collections import defaultdict
import datetime
import math
import os
import os.path as osp
from typing import Optional

from matplotlib import pyplot as plt
from sacred.observers import FileStorageObserver
from stable_baselines import logger as sb_logger
import tensorflow as tf
import tqdm

from imitation.scripts.config.train import train_ex
import imitation.util as util
from imitation.util.trainer import init_trainer


def save(trainer, save_path):
  """Save discriminator and generator."""
  # We implement this here and not in Trainer since we do not want to actually
  # serialize the whole Trainer (including e.g. expert demonstrations).
  trainer.discrim.save(os.path.join(save_path, "discrim"))
  # TODO(gleave): unify this with the saving logic in data_collect?
  # (Needs #43 to be merged before attempting.)
  trainer._gen_policy.save(os.path.join(save_path, "gen_policy"))


@train_ex.main
def train_and_plot(_seed: int,
                   env_name: str,
                   log_dir: str,
                   *,
                   n_epochs: int = 100,
                   n_epochs_per_plot: Optional[float] = None,
                   n_disc_steps_per_epoch: int = 10,
                   n_gen_steps_per_epoch: int = 10000,
                   n_episodes_per_reward_data: int = 5,
                   checkpoint_interval: int = 5,
                   interactive: bool = True,
                   expert_policy=None,
                   init_trainer_kwargs: dict = {},
                   ) -> None:
  """Alternate between training the generator and discriminator.

  Every epoch:
    - Plot discriminator loss during discriminator training steps in blue and
      discriminator loss during generator training steps in red.
    - Plot the performance of the generator policy versus the performance of
      a random policy. Also plot the performance of an expert policy if that is
      provided in the arguments.

  Args:
      _seed: Random seed.
      env_name: The environment to train in, by default 'CartPole-v1'.
      log_dir: Directory to save models and other logging to.
      n_epochs: The number of epochs to train. Each epoch consists of
          `n_disc_steps_per_epoch` discriminator steps followed by
          `n_gen_steps_per_epoch` generator steps.
      n_epochs_per_plot: An optional number, greater than or equal to 1. The
          (possibly fractional) number of epochs between each plot. The first
          plot is at epoch 0, after the first discrim and generator steps.
          If `n_epochs_per_plot is None`, then don't make any plots.
      n_disc_steps_per_epoch: The number of discriminator update steps during
          every training epoch.
      n_gen_plot_episodes: The number of generator update steps during every
          generator epoch.
      n_episodes_per_reward_data: The number of episodes to average over when
          calculating the average episode reward of a policy.
      checkpoint_interval: Save the discriminator and generator models every
          `checkpoint_interval` epochs and after training is complete. If <=0,
          then only save weights after training is complete.
      interactive: Figures are always saved to `output/*.png`. If `interactive`
        is True, then also show plots as they are created.
      expert_policy (BasePolicy or BaseRLModel, optional): If provided, then
          also plot the performance of this expert policy.
      init_trainer_kwargs: Keyword arguments passed to `init_trainer`,
        used to initialize the trainer.
  """
  assert n_epochs_per_plot is None or n_epochs_per_plot >= 1

  with util.make_session():
    trainer = init_trainer(env_name, seed=_seed, log_dir=log_dir,
                           **init_trainer_kwargs)

    tf.logging.info("Logging to %s", log_dir)
    os.makedirs(log_dir, exist_ok=True)
    sb_logger.configure(folder=osp.join(log_dir, 'generator'),
                        format_strs=['tensorboard', 'stdout'])

    plot_idx = 0
    gen_data = ([], [])
    disc_data = ([], [])

    def disc_plot_add_data(gen_mode: bool = False):
      """Evaluates and records the discriminator loss for plotting later.

      Args:
          gen_mode: Whether the generator or the discriminator is active.
              We use this to color the data points.
      """
      nonlocal plot_idx
      mode = "gen" if gen_mode else "dis"
      X, Y = gen_data if gen_mode else disc_data
      # Divide by two since we get two data points (gen and disc) per epoch.
      X.append(plot_idx / 2)
      Y.append(trainer.eval_disc_loss())
      tf.logging.info(
          "plot idx ({}): {} disc loss: {}"
          .format(mode, plot_idx, Y[-1]))
      plot_idx += 1

    def disc_plot_show():
      """Render a plot of discriminator loss vs. training epoch number."""
      plt.scatter(disc_data[0], disc_data[1], c='g', alpha=0.7, s=4,
                  label="discriminator loss (dis step)")
      plt.scatter(gen_data[0], gen_data[1], c='r', alpha=0.7, s=4,
                  label="discriminator loss (gen step)")
      plt.title("Discriminator loss")
      plt.legend()
      _savefig_timestamp("plot_fight_loss_disc", interactive)

    gen_ep_reward = defaultdict(list)
    rand_ep_reward = defaultdict(list)
    exp_ep_reward = defaultdict(list)

    def ep_reward_plot_add_data(env, name):
      """Calculate and record average episode returns."""
      gen_policy = trainer.gen_policy
      gen_ret = util.rollout.mean_return(
          gen_policy, env, n_episodes=n_episodes_per_reward_data)
      gen_ep_reward[name].append(gen_ret)
      tf.logging.info("generator return: {}".format(gen_ret))

      rand_policy = util.make_blank_policy(trainer.env)
      rand_ret = util.rollout.mean_return(
          rand_policy, env, n_episodes=n_episodes_per_reward_data)
      rand_ep_reward[name].append(rand_ret)
      tf.logging.info("random return: {}".format(rand_ret))

      if expert_policy is not None:
          exp_ret = util.rollout.mean_return(
              expert_policy, env, n_episodes=n_episodes_per_reward_data)
          exp_ep_reward[name].append(exp_ret)
          tf.logging.info("exp return: {}".format(exp_ret))

    def ep_reward_plot_show():
      """Render and show average episode reward plots."""
      for name in gen_ep_reward:
        plt.title(name + " Performance")
        plt.xlabel("epochs")
        plt.ylabel("Average reward per episode (n={})"
                   .format(n_episodes_per_reward_data))
        plt.plot(gen_ep_reward[name], label="avg gen ep reward", c="red")
        plt.plot(rand_ep_reward[name],
                 label="avg random ep reward", c="black")
        plt.plot(exp_ep_reward[name], label="avg exp ep reward", c="blue")
        plt.legend()
        _savefig_timestamp("plot_fight_epreward_gen", interactive)

    if n_epochs_per_plot is not None:
      n_plots_per_epoch = 1 / n_epochs_per_plot
    else:
      n_plots_per_epoch = None

    def should_plot_now(epoch) -> bool:
      """For positive epochs, returns True if a plot should be rendered now.

      This also controls the frequency at which `ep_reward_plot_add_data` is
      called, because generating those rollouts is too expensive to perform
      every timestep.
      """
      assert epoch >= 1
      if n_plots_per_epoch is None:
        return False
      plot_num = math.floor(n_plots_per_epoch * epoch)
      prev_plot_num = math.floor(n_plots_per_epoch * (epoch - 1))
      assert abs(plot_num - prev_plot_num) <= 1
      return plot_num != prev_plot_num

    # Collect data for epoch 0.
    if n_epochs_per_plot is not None:
      disc_plot_add_data(False)
      ep_reward_plot_add_data(trainer.env, "Ground Truth Reward")
      ep_reward_plot_add_data(trainer.env_train, "Train Reward")
      ep_reward_plot_add_data(trainer.env_test, "Test Reward")

    for epoch in tqdm.tqdm(range(1, n_epochs+1), desc="epoch"):
      trainer.train_disc(n_disc_steps_per_epoch)
      disc_plot_add_data(False)
      trainer.train_gen(n_gen_steps_per_epoch)
      disc_plot_add_data(True)

      if should_plot_now(epoch):
        disc_plot_show()
        ep_reward_plot_add_data(trainer.env, "Ground Truth Reward")
        ep_reward_plot_add_data(trainer.env_train, "Train Reward")
        ep_reward_plot_add_data(trainer.env_test, "Test Reward")
        ep_reward_plot_show()

      if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
        save(trainer, os.path.join(log_dir, "checkpoints", f"{epoch:05d}"))

    save(trainer, os.path.join(log_dir, "final"))


def _savefig_timestamp(prefix="", also_show=True):
  path = "output/{}_{}.png".format(prefix, datetime.datetime.now())
  plt.savefig(path)
  tf.logging.info("plot saved to {}".format(path))
  if also_show:
    plt.show()


if __name__ == "__main__":
    observer = FileStorageObserver.create(
        osp.join('output', 'sacred', 'train'))
    train_ex.observers.append(observer)
    train_ex.run_commandline()
