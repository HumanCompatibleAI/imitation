"""Train GAIL or AIRL and plot its output.

Can be used as a CLI script, or the `train_and_plot` function can be called
directly.
"""

from collections import defaultdict
import datetime
import math
import os
import os.path as osp
from typing import Dict

from matplotlib import pyplot as plt
from sacred.observers import FileStorageObserver
from stable_baselines import logger as sb_logger
import tensorflow as tf
import tqdm

from imitation.algorithms.adversarial import init_trainer
import imitation.envs.examples  # noqa: F401
from imitation.policies import serialize
from imitation.scripts.config.train_adversarial import train_ex
import imitation.util as util


def save(trainer, save_path):
  """Save discriminator and generator."""
  # We implement this here and not in Trainer since we do not want to actually
  # serialize the whole Trainer (including e.g. expert demonstrations).
  trainer.discrim.save(os.path.join(save_path, "discrim"))
  # TODO(gleave): unify this with the saving logic in data_collect?
  # (Needs #43 to be merged before attempting.)
  serialize.save_stable_model(os.path.join(save_path, "gen_policy"),
                              trainer._gen_policy)


@train_ex.main
def train(_seed: int,
          env_name: str,
          rollout_glob: str,
          log_dir: str,
          *,
          n_epochs: int = 100,
          n_gen_steps_per_epoch: int = 10000,
          n_disc_steps_per_epoch: int = 10,
          init_trainer_kwargs: dict = {},
          n_episodes_eval: int = 50,

          enable_plots: bool = False,
          n_epochs_per_plot: float = 1,
          n_episodes_plot: int = 5,
          expert_policy_plot=None,
          show_plots: bool = True,

          checkpoint_interval: int = 5,
          ) -> Dict[str, float]:
  """Train an adversarial-network-based imitation learning algorithm.

  Plots (turn on using `enable_plots`):
    - Plot discriminator loss during discriminator training steps in blue and
      discriminator loss during generator training steps in red.
    - Plot the performance of the generator policy versus the performance of
      a random policy. Also plot the performance of an expert policy if that is
      provided in the arguments.

  Checkpoints:
    - DiscrimNets are saved to f"{log_dir}/checkpoints/{step}/discrim/",
      where step is either the training epoch or "final".
    - Generator policies are saved to
      f"{log_dir}/checkpoints/{step}/gen_policy/".

  Args:
    _seed: Random seed.
    env_name: The environment to train in.
    log_dir: Directory to save models and other logging to.

    n_epochs: The number of epochs to train. Each epoch consists of
      `n_disc_steps_per_epoch` discriminator steps followed by
      `n_gen_steps_per_epoch` generator steps.
    n_gen_steps_per_epoch: The number of generator update steps during every
      generator epoch.
    n_disc_steps_per_epoch: The number of discriminator update steps during
      every training epoch.

    enable_plots: If True, then enable plotting. If False, then plotting is
      disabled and all the subsequent plot arguments are ignored.
    n_epochs_per_plot: An optional number, greater than or equal to 1. The
      (possibly fractional) number of epochs between each plot. The first
      plot is at epoch 0, after the first discrim and generator steps.
      If `n_epochs_per_plot is None`, then don't make any plots.
    n_episodes_plot: The number of episodes averaged over when
      calculating the average episode reward of a policy for the performance
      plots.
    expert_policy_plot (BasePolicy or BaseRLModel, optional): If provided,
      then also plot the performance of this expert policy.
    show_plots: Figures are always saved to `output/*.png`. If `show_plots`
      is True, then also show plots as they are created.

    n_episodes_eval: The number of episodes to average over when calculating
      the average ground truth reward return of the final policy.
    checkpoint_interval: Save the discriminator and generator models every
      `checkpoint_interval` epochs and after training is complete. If <=0,
      then only save weights after training is complete.
    init_trainer_kwargs: Keyword arguments passed to `init_trainer`,
      used to initialize the trainer.

  Returns:
    results: A dictionary with two keys, "ep_reward_mean" and
      "ep_reward_std_err". The corresponding values are the mean and standard
      error of ground truth episode return for the imitation learning algorithm.
  """
  with util.make_session():
    trainer = init_trainer(env_name, rollout_glob=rollout_glob,
                           seed=_seed, log_dir=log_dir,
                           **init_trainer_kwargs)

    tf.logging.info("Logging to %s", log_dir)
    os.makedirs(log_dir, exist_ok=True)
    sb_logger.configure(folder=osp.join(log_dir, 'generator'),
                        format_strs=['tensorboard', 'stdout'])

    if enable_plots:
      visualizer = _TrainVisualizer(
        trainer, show_plots, n_episodes_plot, n_epochs_per_plot,
        expert_policy_plot)
    else:
      visualizer = None

    # Main training loop.
    for epoch in tqdm.tqdm(range(1, n_epochs+1), desc="epoch"):
      trainer.train_disc(n_disc_steps_per_epoch)
      if enable_plots:
        visualizer.disc_plot_add_data(False)

      trainer.train_gen(n_gen_steps_per_epoch)
      if enable_plots:
        visualizer.disc_plot_add_data(True)

      if enable_plots and visualizer.should_plot_now(epoch):
        visualizer.disc_plot_show()
        visualizer.ep_reward_plot_add_data(trainer.env, "Ground Truth Reward")
        visualizer.ep_reward_plot_add_data(trainer.env_train, "Train Reward")
        visualizer.ep_reward_plot_add_data(trainer.env_test, "Test Reward")
        visualizer.ep_reward_plot_show()

      if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
        save(trainer, os.path.join(log_dir, "checkpoints", f"{epoch:05d}"))

    # Save final artifacts.
    save(trainer, os.path.join(log_dir, "checkpoints", "final"))

    # Final evaluation of imitation policy.
    stats = util.rollout.rollout_stats(trainer.gen_policy,
                                       trainer.env,
                                       n_episodes=n_episodes_eval)
    assert stats["n_traj"] >= n_episodes_eval
    ep_reward_mean = stats["return_mean"]
    ep_reward_std_err = stats["return_std"] / math.sqrt(n_episodes_eval)
    print("[result] Mean Episode Return: "
          f"{ep_reward_mean:.4g} ± {ep_reward_std_err:.3g} "
          f"(n={stats['n_traj']})")

    return dict(ep_reward_mean=ep_reward_mean,
                ep_reward_std_err=ep_reward_std_err)


class _TrainVisualizer:
  def __init__(self,
               trainer: "imitation.algorithms.adversarial.AdversarialTrainer",
               show_plots: bool,
               n_episodes_per_reward_data: int,
               n_epochs_per_plot: float,
               expert_policy=None):
    """
    Args:
      trainer: AdversarialTrainer used to perform rollouts.
      show_plots: If True, then `plt.show()` plot in addition to saving them.
      n_episodes_per_reward_data: The number of episodes to rollout out when
        determining the mean episode reward.
      n_epochs_per_plot: An optional number, greater than or equal to 1. The
        (possibly fractional) number of epochs between each plot. The first
        plot is at epoch 0, after the first discrim and generator steps.
        If `n_epochs_per_plot is None`, then don't make any plots.
      expert_policy (BasePolicy or BaseRLModel, optional): If provided,
          then also plot the performance of this expert policy.
    """
    self.trainer = trainer
    self.show_plots = show_plots
    self.n_episodes_per_reward_data = n_episodes_per_reward_data
    self.expert_policy = expert_policy
    self.plot_idx = 0
    assert n_epochs_per_plot >= 1
    self.n_epochs_per_plot = n_epochs_per_plot
    self.gen_data = ([], [])
    self.disc_data = ([], [])

    self.gen_ep_reward = defaultdict(list)
    self.rand_ep_reward = defaultdict(list)
    self.exp_ep_reward = defaultdict(list)

    # Collect data for epoch 0.
    self.disc_plot_add_data(False)
    self.ep_reward_plot_add_data(self.trainer.env, "Ground Truth Reward")
    self.ep_reward_plot_add_data(self.trainer.env_train, "Train Reward")
    self.ep_reward_plot_add_data(self.trainer.env_test, "Test Reward")

  def should_plot_now(self, epoch: int) -> bool:
    """For positive epochs, returns True if a plot should be rendered now.

    This also controls the frequency at which `ep_reward_plot_add_data` is
    called, because generating those rollouts is too expensive to perform
    every timestep.
    """
    assert epoch >= 1
    plot_num = math.floor(epoch / self.n_epochs_per_plot)
    prev_plot_num = math.floor((epoch - 1) / self.n_epochs_per_plot)
    assert abs(plot_num - prev_plot_num) <= 1
    return plot_num != prev_plot_num

  def disc_plot_add_data(self, gen_mode: bool = False):
    """Evaluates and records the discriminator loss for plotting later.

    Args:
        gen_mode: Whether the generator or the discriminator is active.
            We use this to color the data points.
    """
    mode = "gen" if gen_mode else "dis"
    X, Y = self.gen_data if gen_mode else self.disc_data
    # Divide by two since we get two data points (gen and disc) per epoch.
    X.append(self.plot_idx / 2)
    Y.append(self.trainer.eval_disc_loss())
    tf.logging.info(
        "plot idx ({}): {} disc loss: {}"
        .format(mode, self.plot_idx, Y[-1]))
    self.plot_idx += 1

  def disc_plot_show(self):
    """Render a plot of discriminator loss vs. training epoch number."""
    plt.scatter(self.disc_data[0], self.disc_data[1], c='g', alpha=0.7, s=4,
                label="discriminator loss (dis step)")
    plt.scatter(self.gen_data[0], self.gen_data[1], c='r', alpha=0.7, s=4,
                label="discriminator loss (gen step)")
    plt.title("Discriminator loss")
    plt.legend()
    _savefig_timestamp("plot_fight_loss_disc", self.show_plots)

  def ep_reward_plot_add_data(self, env, name):
    """Calculate and record average episode returns."""
    gen_policy = self.trainer.gen_policy
    gen_ret = util.rollout.mean_return(
        gen_policy, env, n_episodes=self.n_episodes_per_reward_data)
    self.gen_ep_reward[name].append(gen_ret)
    tf.logging.info("generator return: {}".format(gen_ret))

    rand_policy = util.init_rl(self.trainer.env)
    rand_ret = util.rollout.mean_return(
        rand_policy, env, n_episodes=self.n_episodes_per_reward_data)
    self.rand_ep_reward[name].append(rand_ret)
    tf.logging.info("random return: {}".format(rand_ret))

    if self.expert_policy is not None:
      exp_ret = util.rollout.mean_return(
          self.expert_policy, env, n_episodes=self.n_episodes_per_reward_data)
      self.exp_ep_reward[name].append(exp_ret)
      tf.logging.info("exp return: {}".format(exp_ret))

  def ep_reward_plot_show(self):
    """Render and show average episode reward plots."""
    for name in self.gen_ep_reward:
      plt.title(name + " Performance")
      plt.xlabel("epochs")
      plt.ylabel("Average reward per episode (n={})"
                 .format(self.n_episodes_per_reward_data))
      plt.plot(self.gen_ep_reward[name], label="avg gen ep reward", c="red")
      plt.plot(self.rand_ep_reward[name],
               label="avg random ep reward", c="black")
      if self.expert_policy is not None:
        plt.plot(self.exp_ep_reward[name], label="avg exp ep reward", c="blue")
      plt.legend()
      _savefig_timestamp("plot_fight_epreward_gen", self.show_plots)


def _savefig_timestamp(prefix="", also_show=True):
  path = "output/{}_{}.png".format(prefix, datetime.datetime.now())
  plt.savefig(path)
  tf.logging.info("plot saved to {}".format(path))
  if also_show:
    plt.show()


def main_console():
  observer = FileStorageObserver.create(osp.join('output', 'sacred', 'train'))
  train_ex.observers.append(observer)
  train_ex.run_commandline()


if __name__ == "__main__":
  main_console()
