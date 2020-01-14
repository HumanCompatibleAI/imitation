"""Train GAIL or AIRL and plot its output.

Can be used as a CLI script, or the `train_and_plot` function can be called
directly.
"""

from collections import defaultdict
import os
import os.path as osp
import pickle
from typing import Optional

from matplotlib import pyplot as plt
import numpy as np
from sacred.observers import FileStorageObserver
import tensorflow as tf
import tqdm

from imitation.algorithms.adversarial import AdversarialTrainer, init_trainer
import imitation.envs.examples  # noqa: F401
from imitation.policies import serialize
from imitation.scripts.config.train_adversarial import train_ex
import imitation.util as util
import imitation.util.sacred as sacred_util


def save(trainer, save_path):
  """Save discriminator and generator."""
  # We implement this here and not in Trainer since we do not want to actually
  # serialize the whole Trainer (including e.g. expert demonstrations).
  trainer.discrim.save(os.path.join(save_path, "discrim"))
  # TODO(gleave): unify this with the saving logic in data_collect?
  # (Needs #43 to be merged before attempting.)
  serialize.save_stable_model(os.path.join(save_path, "gen_policy"),
                              trainer.gen_policy)


@train_ex.main
def train(_run,
          _seed: int,
          env_name: str,
          rollout_path: str,
          n_expert_demos: Optional[int],
          log_dir: str,
          init_trainer_kwargs: dict,
          total_timesteps: int,
          n_episodes_eval: int,

          plot_interval: int,
          n_plot_episodes: int,
          extra_episode_data_interval: int,
          show_plots: bool,
          init_tensorboard: bool,

          checkpoint_interval: int,
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
    rollout_path: Path to pickle containing list of Trajectories. Used as
      expert demonstrations.
    n_expert_demos: The number of expert trajectories to actually use
      after loading them from `rollout_path`.
      If None, then use all available trajectories.
      If `n_expert_demos` is an `int`, then use exactly `n_expert_demos`
      trajectories, erroring if there aren't enough trajectories. If there are
      surplus trajectories, then use the
      first `n_expert_demos` trajectories and drop the rest.
    log_dir: Directory to save models and other logging to.

    init_trainer_kwargs: Keyword arguments passed to `init_trainer`,
      used to initialize the trainer.
    total_timesteps: The number of transitions to sample from the environment
      during training.
    n_episodes_eval: The number of episodes to average over when calculating
      the average episode reward of the imitation policy for return.

    plot_interval: The number of epochs between each plot. If negative,
      then plots are disabled. If zero, then only plot at the end of training.
    n_plot_episodes: The number of episodes averaged over when
      calculating the average episode reward of a policy for the performance
      plots.
    extra_episode_data_interval: Usually mean episode rewards are calculated
      immediately before every plot. Set this parameter to a nonnegative number
      to also add episode reward data points every
      `extra_episodes_data_interval` epochs.
    show_plots: Figures are always saved to `f"{log_dir}/plots/*.png"`. If
      `show_plots` is True, then also show plots as they are created.
    init_tensorboard: If True, then write tensorboard logs to `{log_dir}/sb_tb`.

    checkpoint_interval: Save the discriminator and generator models every
      `checkpoint_interval` epochs and after training is complete. If <=0,
      then only save weights after training is complete.

  Returns:
    A dictionary with two keys. "imit_stats" gives the return value of
      `rollout_stats()` on rollouts test-reward-wrapped
      environment, using the final policy (remember that the ground-truth reward
      can be recovered from the "monitor_return" key). "expert_stats" gives the
      return value of `rollout_stats()` on the expert demonstrations loaded from
      `rollout_path`.
  """
  total_timesteps = int(total_timesteps)

  tf.logging.info("Logging to %s", log_dir)
  os.makedirs(log_dir, exist_ok=True)
  sacred_util.build_sacred_symlink(log_dir, _run)

  # Calculate stats for expert rollouts. Used for plot and return value.
  with open(rollout_path, "rb") as f:
    expert_trajs = pickle.load(f)

  if n_expert_demos is not None:
    assert len(expert_trajs) >= n_expert_demos
    expert_trajs = expert_trajs[:n_expert_demos]

  expert_stats = util.rollout.rollout_stats(expert_trajs)

  with util.make_session():
    if init_tensorboard:
      sb_tensorboard_dir = osp.join(log_dir, "sb_tb")
      kwargs = init_trainer_kwargs
      kwargs["init_rl_kwargs"] = kwargs.get("init_rl_kwargs", {})
      kwargs["init_rl_kwargs"]["tensorboard_log"] = sb_tensorboard_dir

    trainer = init_trainer(env_name, expert_trajs,
                           seed=_seed, log_dir=log_dir,
                           **init_trainer_kwargs)

    if plot_interval >= 0:
      visualizer = _TrainVisualizer(
        trainer=trainer,
        show_plots=show_plots,
        n_episodes_per_reward_data=n_plot_episodes,
        log_dir=log_dir,
        expert_mean_ep_reward=expert_stats["return_mean"])
    else:
      visualizer = None

    # Main training loop.
    n_epochs = total_timesteps // trainer.gen_batch_size
    assert n_epochs >= 1, ("No updates (need at least "
                           f"{trainer.gen_batch_size} timesteps, have only "
                           f"total_timesteps={total_timesteps})!")

    for epoch in tqdm.tqdm(range(1, n_epochs+1), desc="epoch"):
      trainer.train_gen(trainer.gen_batch_size)
      if visualizer:
        visualizer.add_data_disc_loss(True, epoch)
      trainer.train_disc(trainer.disc_batch_size)

      util.logger.dumpkvs()

      if visualizer:
        visualizer.add_data_disc_loss(False, epoch)

        if (extra_episode_data_interval > 0
            and epoch % extra_episode_data_interval == 0):  # noqa: E129
          visualizer.add_data_ep_reward(epoch)

        if plot_interval > 0 and epoch % plot_interval == 0:
          visualizer.plot_disc_loss()
          visualizer.add_data_ep_reward(epoch)
          # Add episode mean rewards only at plot time because it is expensive.
          visualizer.plot_ep_reward()

      if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
        save(trainer, os.path.join(log_dir, "checkpoints", f"{epoch:05d}"))

    # Save final artifacts.
    save(trainer, os.path.join(log_dir, "checkpoints", "final"))

    if visualizer:
      visualizer.plot_disc_loss()
      visualizer.add_data_ep_reward(epoch)
      visualizer.plot_ep_reward()

    # Final evaluation of imitation policy.
    results = {}
    sample_until_eval = util.rollout.min_episodes(n_episodes_eval)
    trajs = util.rollout.generate_trajectories(trainer.gen_policy,
                                               trainer.venv_test,
                                               sample_until=sample_until_eval)
    results["imit_stats"] = util.rollout.rollout_stats(trajs)
    results["expert_stats"] = expert_stats

    return results


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

    self.venv_norm_obs = util.reapply_vec_normalize(
      trainer.venv, trainer.venv_train_norm, disable_norm_reward=True)
    self.plot_idx = 0
    self.gen_data = ([], [])
    self.disc_data = ([], [])

    self.ep_reward_X = []
    self.gen_ep_reward = defaultdict(list)
    self.rand_ep_reward = defaultdict(list)

    # Collect data for epoch 0.
    self.add_data_disc_loss(False, 0)
    self.add_data_ep_reward(0)

  def add_data_disc_loss(self, generator_active: bool, epoch: int):
    """Evaluates and records the discriminator loss for plotting later.

    Args:
        generator_active: True if the generator is being trained. Otherwise, the
            discriminator is being trained.  We use this to color the data
            points.
    """
    if not generator_active and len(self.gen_data[0]) == 0:
      # Skip because `eval_disc_loss()` doesn't have generator samples
      # available. (`trainer.train_gen()` hasn't been called yet)
      return
    mode = "gen" if generator_active else "dis"
    X, Y = self.gen_data if generator_active else self.disc_data
    if not generator_active:
      X.append(epoch)
    else:
      X.append(epoch + 0.5)
    Y.append(self.trainer.eval_disc_loss())
    tf.logging.info("epoch ({}): {} disc loss: {}".format(mode, epoch, Y[-1]))

  def plot_disc_loss(self):
    """Render a plot of discriminator loss vs. training epoch number."""
    plt.scatter(self.disc_data[0], self.disc_data[1], c='g', alpha=0.7, s=4,
                label="discriminator loss (dis step)")
    plt.scatter(self.gen_data[0], self.gen_data[1], c='r', alpha=0.7, s=4,
                label="discriminator loss (gen step)")
    plt.title("Discriminator loss")
    plt.legend()
    self._savefig("plot_fight_loss_disc", self.show_plots)

  def add_data_ep_reward(self, epoch):
    """Calculate and record average episode returns."""
    if epoch in self.ep_reward_X:
      # Don't calculate ep reward twice.
      return
    self.ep_reward_X.append(epoch)

    gen_policy = self.trainer.gen_policy
    rand_policy = util.init_rl(self.trainer.venv)
    sample_until = util.rollout.min_episodes(self.n_episodes_per_reward_data)
    trajs_rand = util.rollout.generate_trajectories(
      rand_policy, self.venv_norm_obs, sample_until)
    trajs_gen = util.rollout.generate_trajectories(
      gen_policy, self.venv_norm_obs, sample_until)

    for reward_fn, reward_name in [(None, "Ground Truth Reward"),
                                   (self.trainer.reward_train, "Train Reward"),
                                   (self.trainer.reward_test, "Test Reward")]:
      if reward_fn is None:
        trajs_rand_rets = [np.sum(traj.rews) for traj in trajs_rand]
        trajs_gen_rets = [np.sum(traj.rews) for traj in trajs_gen]
      else:
        trajs_rand_rets = [
            np.sum(util.rollout.recalc_rewards_traj(traj, reward_fn))
            for traj in trajs_rand]
        trajs_gen_rets = [
            np.sum(util.rollout.recalc_rewards_traj(traj, reward_fn))
            for traj in trajs_gen]

      gen_ret = np.mean(trajs_gen_rets)
      rand_ret = np.mean(trajs_rand_rets)
      self.gen_ep_reward[reward_name].append(gen_ret)
      self.rand_ep_reward[reward_name].append(rand_ret)
      tf.logging.info(f"{reward_name} generator return: {gen_ret}")
      tf.logging.info(f"{reward_name} random return: {rand_ret}")

  def plot_ep_reward(self):
    """Render and show average episode reward plots."""
    for name in self.gen_ep_reward:
      plt.title(name + " Performance")
      plt.xlabel("epochs")
      plt.ylabel("Average reward per episode (n={})"
                 .format(self.n_episodes_per_reward_data))
      X = self.ep_reward_X
      plt.plot(X, self.gen_ep_reward[name], label="avg gen ep reward", c="red")
      plt.plot(X, self.rand_ep_reward[name],
               label="avg random ep reward", c="black")

      name = name.lower().replace(' ', '-')
      if (self.expert_mean_ep_reward is not None and
              name == "ground-truth-reward"):
          plt.axhline(y=self.expert_mean_ep_reward,
                      linestyle='dashed',
                      label=f"expert (return={self.expert_mean_ep_reward:.2g})",
                      color="black")
      plt.legend()
      self._savefig(f"plot_fight_epreward_gen_{name}", self.show_plots)

  def _savefig(self, prefix="", also_show=True):
    plot_dir = osp.join(self.log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    path = osp.join(plot_dir, f"{prefix}_{self.plot_idx}")
    plt.savefig(path)
    tf.logging.info("plot saved to {}".format(path))
    if also_show:
      plt.show()
    plt.clf()


def main_console():
  observer = FileStorageObserver.create(osp.join('output', 'sacred', 'train'))
  train_ex.observers.append(observer)
  train_ex.run_commandline()


if __name__ == "__main__":
  main_console()
