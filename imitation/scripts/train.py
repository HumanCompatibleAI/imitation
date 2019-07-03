"""Train an IRL algorithm and plot its output.

Can be used as a CLI script, or the `train_and_plot` function can be called
directly."""

import argparse
from collections import defaultdict
import datetime
import math
import os
from typing import Optional, Union

import gin
import gin.tf
import gym
from matplotlib import pyplot as plt
import tensorflow as tf
import tqdm

from imitation.airl import AIRLTrainer
import imitation.util as util
from imitation.util.trainer import init_trainer


@gin.configurable
def train_and_plot(policy_dir: str = "expert_models",
                   env: Union[gym.Env, str] = 'CartPole-v1',
                   *,
                   n_epochs: int = 100,
                   n_epochs_per_plot: Optional[float] = None,
                   n_disc_steps_per_epoch: int = 10,
                   n_gen_steps_per_epoch: int = 10000,
                   n_episodes_per_reward_data: int = 5,
                   interactive: bool = True,
                   trainer: Optional[AIRLTrainer] = None,
                   **airl_trainer_kwargs
                   ) -> None:
  """Alternate between training the generator and discriminator.

  Every epoch:
    - Plot discriminator loss during discriminator training steps in blue and
      discriminator loss during generator training steps in red.
    - Plot the performance of the generator policy versus the performance of
      a random policy.

  Args:
      policy_dir: Path to a directory that holds pickled expert policies.
      env: The environment to train in, by default 'CartPole-v1'. This
          argument is ignored if the `trainer` is manually provided via the
          `trainer` argument.
      n_epochs: The number of epochs to train. Each epoch consists of
          `n_disc_steps_per_epoch` discriminator steps followed by
          `n_gen_steps_per_epoch` generator steps.
      n_epochs_per_plot: An optional number, greater than or equal to 1. The
          (possibly fractional)
          number of epochs between each plot. The first plot is at epoch 0,
          right after the first discrim and generator steps.

          If `n_epochs_per_plot is None`, then don't make any plots.
      n_disc_steps_per_epoch: The number of discriminator update steps during
          every training epoch.
      n_gen_plot_episodes: The number of generator update steps during every
          generator epoch.
      n_episodes_per_reward_data: The number of episodes to average over when
          calculating the average episode reward of a policy.
      interactive: Figures are always saved to "output/*.png". If `interactive`
        is True, then also show plots as they are created.
      trainer: If this is provided, then start training on this AIRLTrainer
        instead of initializing a new one using `airl_trainer_kwargs`. Also,
        ignore the `env` argument.
  """
  assert n_epochs_per_plot is None or n_epochs_per_plot >= 1
  if trainer is None:
    assert env is not None
    trainer = init_trainer(env, policy_dir=policy_dir,
                           **airl_trainer_kwargs)
  env = trainer.env

  os.makedirs("output/", exist_ok=True)

  plot_idx = 0
  gen_data = ([], [])
  disc_data = ([], [])

  def disc_plot_add_data(gen_mode=False):
    """ Evaluate and record the discriminator loss for plotting later.

    Args:
        gen_mode (bool): Whether the generator or the discriminator is active.
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
    """Calculate and record the average episode reward from rollouts of env."""
    gen_policy = trainer.gen_policy
    rand_policy = util.make_blank_policy(env)
    exp_policy = trainer.expert_policies[-1]

    gen_rew = util.rollout.total_reward(
        gen_policy, env, n_episodes=n_episodes_per_reward_data
    ) / n_episodes_per_reward_data
    rand_rew = util.rollout.total_reward(
        rand_policy, env, n_episodes=n_episodes_per_reward_data
    ) / n_episodes_per_reward_data
    exp_rew = util.rollout.total_reward(
        exp_policy, env, n_episodes=n_episodes_per_reward_data
    ) / n_episodes_per_reward_data
    gen_ep_reward[name].append(gen_rew)
    rand_ep_reward[name].append(rand_rew)
    exp_ep_reward[name].append(exp_rew)
    tf.logging.info("generator reward: {}".format(gen_rew))
    tf.logging.info("random reward: {}".format(rand_rew))
    tf.logging.info("exp reward: {}".format(exp_rew))

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
    ep_reward_plot_add_data(env, "True Reward")
    ep_reward_plot_add_data(trainer.env_wrapped_test, "Learned Reward")

  for epoch in tqdm.tqdm(range(1, n_epochs+1), desc="epoch"):
    trainer.train_disc(n_disc_steps_per_epoch)
    disc_plot_add_data(False)
    trainer.train_gen(n_gen_steps_per_epoch)
    disc_plot_add_data(True)

    if should_plot_now(epoch):
      disc_plot_show()
      ep_reward_plot_add_data(env, "True Reward")
      ep_reward_plot_add_data(trainer.env_wrapped_test, "Learned Reward")
      ep_reward_plot_show()

  return trainer, gen_data, disc_data, gen_ep_reward


def _savefig_timestamp(prefix="", also_show=True):
  path = "output/{}_{}.png".format(prefix, datetime.datetime.now())
  plt.savefig(path)
  tf.logging.info("plot saved to {}".format(path))
  if also_show:
    plt.show()


def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  train_and_plot()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gin_config",
                      default='configs/cartpole_orig_airl_repro.gin')
  args = parser.parse_args()

  gin.parse_config_file(args.gin_config)

  main()
