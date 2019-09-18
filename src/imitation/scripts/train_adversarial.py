"""Train GAIL or AIRL and plot its output.

Can be used as a CLI script, or the `train_and_plot` function can be called
directly.
"""

from collections import defaultdict
import os
import os.path as osp
from typing import Optional

from matplotlib import pyplot as plt
from sacred.observers import FileStorageObserver
from stable_baselines import logger as sb_logger
import tensorflow as tf
import tqdm

from imitation.algorithms.adversarial import AdversarialTrainer, init_trainer
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
          n_epochs: int,
          n_gen_steps_per_epoch: int,
          n_disc_steps_per_epoch: int,
          init_trainer_kwargs: dict,
          n_episodes_eval: int,

          plot_interval: int,
          n_plot_episodes: int,
          show_plots: bool,

          checkpoint_interval: int = 5,
          ) -> dict:
  """Train an adversarial-network-based imitation learning algorithm.

  Plots (turn on using `plot_interval > 0`):
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
    rollout_glob: A bash-style regex pattern from which rollout pickles are
      loaded.
    log_dir: Directory to save models and other logging to.

    n_epochs: The number of epochs to train. Each epoch consists of
      `n_disc_steps_per_epoch` discriminator steps followed by
      `n_gen_steps_per_epoch` generator steps.
    n_gen_steps_per_epoch: The number of generator update steps during every
      training epoch.
    n_disc_steps_per_epoch: The number of discriminator update steps during
      every training epoch.
    init_trainer_kwargs: Keyword arguments passed to `init_trainer`,
      used to initialize the trainer.
    n_episodes_eval: The number of episodes to average over when calculating
      the average episode reward of the imitation policy for return.

    plot_interval: The number of epochs between each plot. (If nonpositive,
      then plots are disabled).
    n_plot_episodes: The number of episodes averaged over when
      calculating the average episode reward of a policy for the performance
      plots.
    show_plots: Figures are always saved to `f"{log_dir}/plots/*.png"`. If
      `show_plots` is True, then also show plots as they are created.

    checkpoint_interval: Save the discriminator and generator models every
      `checkpoint_interval` epochs and after training is complete. If <=0,
      then only save weights after training is complete.

  Returns:
    A dictionary with the following keys: "rollout_stats" (return value of
      `rollout_stats()` on the test reward wrapped environment),
      "log_dir", "transfer_reward_path", "transfer_reward_type".
  """
  with util.make_session():
    trainer = init_trainer(env_name, rollout_glob=rollout_glob,
                           seed=_seed, log_dir=log_dir,
                           **init_trainer_kwargs)

    tf.logging.info("Logging to %s", log_dir)
    os.makedirs(log_dir, exist_ok=True)
    sb_logger.configure(folder=osp.join(log_dir, 'generator'),
                        format_strs=['tensorboard', 'stdout'])

    if plot_interval > 0:
      # If `n_expert_demos` was provided, then we have enough information
      # to determine the expert's mean episode reward just from inspecting
      # the `trainer.expert_demos` (Transitions). Kind of sad that there isn't
      # a clean way to pull the raw `List[Trajectory]` out of `init_trainer`.
      n_expert_demos = init_trainer_kwargs.get("n_expert_demos")
      if n_expert_demos is not None:
        expert_mean_ep_reward = trainer.expert_demos.rews / n_expert_demos
      else:
        expert_mean_ep_reward = None

      visualizer = _TrainVisualizer(
        trainer=trainer,
        show_plots=show_plots,
        n_episodes_per_reward_data=n_plot_episodes,
        log_dir=log_dir,
        expert_mean_ep_reward=expert_mean_ep_reward)
    else:
      visualizer = None

    # Main training loop.
    for epoch in tqdm.tqdm(range(1, n_epochs+1), desc="epoch"):
      trainer.train_disc(n_disc_steps_per_epoch)
      if visualizer:
        visualizer.add_data_disc_loss(False)

      trainer.train_gen(n_gen_steps_per_epoch)
      if visualizer:
        visualizer.add_data_disc_loss(True)

      if visualizer and epoch % plot_interval == 0:
        visualizer.plot_disc_loss()
        visualizer.add_data_ep_reward(trainer.env, "Ground Truth Reward")
        visualizer.add_data_ep_reward(trainer.env_train, "Train Reward")
        visualizer.add_data_ep_reward(trainer.env_test, "Test Reward")
        visualizer.plot_ep_reward()

      if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
        save(trainer, os.path.join(log_dir, "checkpoints", f"{epoch:05d}"))

    # Save final artifacts.
    save(trainer, os.path.join(log_dir, "checkpoints", "final"))

    # Final evaluation of imitation policy.
    sample_until_eval = util.rollout.min_episodes(n_episodes_eval)
    stats = util.rollout.rollout_stats(trainer.gen_policy,
                                       trainer.env_test,
                                       sample_until=sample_until_eval)
    assert stats["n_traj"] >= n_episodes_eval

    reward_path = os.path.join(log_dir, "checkpoints", "final", "discrim")

    return dict(rollout_stats=stats,
                log_dir=log_dir,
                transfer_reward_path=reward_path,
                transfer_reward_type="DiscrimNet")


class _TrainVisualizer:
  def __init__(self,
               trainer: AdversarialTrainer,
               show_plots: bool,
               n_episodes_per_reward_data: int,
               log_dir: str,
               expert_mean_ep_reward: Optional[float] = None):
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
      expert_mean_ep_reward: If provided, then also plot the performance of
        the expert policy.
    """
    self.trainer = trainer
    self.show_plots = show_plots
    self.n_episodes_per_reward_data = n_episodes_per_reward_data
    self.log_dir = log_dir
    self.expert_mean_ep_reward = expert_mean_ep_reward
    self.plot_idx = 0
    self.gen_data = ([], [])
    self.disc_data = ([], [])

    self.gen_ep_reward = defaultdict(list)
    self.rand_ep_reward = defaultdict(list)

    # Collect data for epoch 0.
    self.add_data_disc_loss(False)
    self.add_data_ep_reward(self.trainer.env, "Ground Truth Reward")
    self.add_data_ep_reward(self.trainer.env_train, "Train Reward")
    self.add_data_ep_reward(self.trainer.env_test, "Test Reward")

  def add_data_disc_loss(self, generator_active: bool = False):
    """Evaluates and records the discriminator loss for plotting later.

    Args:
        generator_active: True if the generator is being trained. Otherwise, the
            discriminator is being trained.  We use this to color the data
            points.
    """
    mode = "gen" if generator_active else "dis"
    X, Y = self.gen_data if generator_active else self.disc_data
    # Divide by two since we get two data points (gen and disc) per epoch.
    X.append(self.plot_idx / 2)
    Y.append(self.trainer.eval_disc_loss())
    tf.logging.info(
        "plot idx ({}): {} disc loss: {}"
        .format(mode, self.plot_idx, Y[-1]))
    self.plot_idx += 1

  def plot_disc_loss(self):
    """Render a plot of discriminator loss vs. training epoch number."""
    plt.scatter(self.disc_data[0], self.disc_data[1], c='g', alpha=0.7, s=4,
                label="discriminator loss (dis step)")
    plt.scatter(self.gen_data[0], self.gen_data[1], c='r', alpha=0.7, s=4,
                label="discriminator loss (gen step)")
    plt.title("Discriminator loss")
    plt.legend()
    self._savefig("plot_fight_loss_disc", self.show_plots)

  def add_data_ep_reward(self, env, name):
    """Calculate and record average episode returns."""
    sample_until = util.rollout.min_episodes(self.n_episodes_per_reward_data)

    gen_policy = self.trainer.gen_policy
    gen_ret = util.rollout.mean_return(gen_policy, env, sample_until)
    self.gen_ep_reward[name].append(gen_ret)
    tf.logging.info("generator return: {}".format(gen_ret))

    rand_policy = util.init_rl(self.trainer.env)
    rand_ret = util.rollout.mean_return(rand_policy, env, sample_until)
    self.rand_ep_reward[name].append(rand_ret)
    tf.logging.info("random return: {}".format(rand_ret))

  def plot_ep_reward(self):
    """Render and show average episode reward plots."""
    for name in self.gen_ep_reward:
      plt.title(name + " Performance")
      plt.xlabel("epochs")
      plt.ylabel("Average reward per episode (n={})"
                 .format(self.n_episodes_per_reward_data))
      plt.plot(self.gen_ep_reward[name], label="avg gen ep reward", c="red")
      plt.plot(self.rand_ep_reward[name],
               label="avg random ep reward", c="black")
      if self.expert_mean_ep_reward is not None:
        plt.hlines(y=self.expert_mean_ep_reward,
                   linestyles='dashed',
                   label="expert (return={self.expert_mean_ep_reward:.2g})",
                   c="black")
      plt.legend()
      self._savefig("plot_fight_epreward_gen", self.show_plots)

  def _savefig(self, prefix="", also_show=True):
    plot_dir = osp.join(self.log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    path = osp.join(plot_dir, f"{prefix}_{self.plot_idx}")
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
