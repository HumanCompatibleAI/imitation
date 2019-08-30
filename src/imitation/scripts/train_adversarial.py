"""Train GAIL or AIRL and plot its output.

Can be used as a CLI script, or the `train_and_plot` function can be called
directly.
"""

from collections import defaultdict
import math
import os
import os.path as osp
from warning import warn

from matplotlib import pyplot as plt
import ray.tune
from sacred.observers import FileStorageObserver
from stable_baselines import logger as sb_logger
import tensorflow as tf
import tqdm

from imitation.algorithms.adversarial import init_trainer
import imitation.envs.examples  # noqa: F401
from imitation.policies import serialize
from imitation.rewards.discrim_net import DiscrimNetAIRL, DiscrimNetGAIL
from imitation.scripts.config.train_adversarial import train_ex
from imitation.scripts.util.multi import ray_tune_active
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

          plot_interval: int = -1,
          n_plot_episodes: int = 5,
          expert_policy_plot=None,
          show_plots: bool = True,

          ray_tune_interval: int = -1,

          checkpoint_interval: int = 5,
          ) -> dict:
  """Train an adversarial-network-based imitation learning algorithm.

  Plots (turn on using `plot_interval > 0`):
    - Plot discriminator loss during discriminator training steps in blue and
      discriminator loss during generator training steps in red.
    - Plot the performance of the generator policy versus the performance of
      a random policy. Also plot the performance of an expert policy if that is
      provided in the arguments.

  Ray Tune (turn on using `ray_tune_interval > 0`):
    - Track the episode reward mean of the imitation policy by performing
      rollouts every `ray_tune_interval` epochs.

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

    plot_interval: The number of epochs between each plot. (If nonpositive,
      then plots are disabled).
    n_episodes_plot: The number of episodes averaged over when
      calculating the average episode reward of a policy for the performance
      plots.
    expert_policy_plot (BasePolicy or BaseRLModel, optional): If provided,
      then also plot the performance of this expert policy.
    show_plots: Figures are always saved to `output/*.png`. If `show_plots`
      is True, then also show plots as they are created.

    ray_tune_interval: The number of epochs between calls to `ray.tune.track`.
      If nonpositive, disables ray tune. Otherwise, enables hooks
      that call `ray.tune.track` to track the imitation policy's mean episode
      reward over time. The script will crash unless `ray.tune` was
      externally initialized.

    n_episodes_eval: The number of episodes to average over when calculating
      the average ground truth reward return of the imitation policy for return
      and for `ray.tune.track`.
    checkpoint_interval: Save the discriminator and generator models every
      `checkpoint_interval` epochs and after training is complete. If <=0,
      then only save weights after training is complete.
    init_trainer_kwargs: Keyword arguments passed to `init_trainer`,
      used to initialize the trainer.

  Returns:
    A dictionary with the following keys: "ep_reward_mean" and
    "ep_reward_std_err", "log_dir", "transfer_reward_path",
    "transfer_reward_type".
  """
  if ray_tune_interval <= 0 and ray_tune_active():
    warn("This Sacred run isn't configured for Ray Tune "
         "even though Ray Tune is active!")

  with util.make_session():
    trainer = init_trainer(env_name, rollout_glob=rollout_glob,
                           seed=_seed, log_dir=log_dir,
                           **init_trainer_kwargs)

    tf.logging.info("Logging to %s", log_dir)
    os.makedirs(log_dir, exist_ok=True)
    sb_logger.configure(folder=osp.join(log_dir, 'generator'),
                        format_strs=['tensorboard', 'stdout'])

    if plot_interval > 0:
      visualizer = _TrainVisualizer(
        trainer=trainer,
        show_plots=show_plots,
        n_episodes_per_reward_data=n_plot_episodes,
        log_dir=log_dir,
        expert_policy=expert_policy_plot)
    else:
      visualizer = None

    # Main training loop.
    for epoch in tqdm.tqdm(range(1, n_epochs+1), desc="epoch"):
      trainer.train_disc(n_disc_steps_per_epoch)
      if visualizer:
        visualizer.disc_plot_add_data(False)

      trainer.train_gen(n_gen_steps_per_epoch)
      if visualizer:
        visualizer.disc_plot_add_data(True)

      if visualizer and epoch % plot_interval == 0:
        visualizer.disc_plot_show()
        visualizer.ep_reward_plot_add_data(trainer.env, "Ground Truth Reward")
        visualizer.ep_reward_plot_add_data(trainer.env_train, "Train Reward")
        visualizer.ep_reward_plot_add_data(trainer.env_test, "Test Reward")
        visualizer.ep_reward_plot_show()

      if ray_tune_interval > 0 and epoch % ray_tune_interval == 0:
        gen_ret = util.rollout.mean_return(
            trainer.gen_policy, trainer.env, n_episodes=n_episodes_eval)
        ray.tune.track.log(episode_reward_mean=gen_ret)

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

    reward_path = os.path.join(log_dir, "checkpoints", "final", "discrim")
    # TODO(shwang): I think Serializable should store the save_type, and
    # Serializable.save() should return save_type.
    if isinstance(trainer.discrim, DiscrimNetAIRL):
      reward_type = "DiscrimNetAIRL"
    elif isinstance(trainer.discrim, DiscrimNetGAIL):
      reward_type = "DiscrimNetGAIL"
    else:
      raise RuntimeError(f"Unknown reward type for {trainer.discrim}")

    return dict(ep_reward_mean=ep_reward_mean,
                ep_reward_std_err=ep_reward_std_err,
                log_dir=log_dir,
                transfer_reward_path=reward_path,
                transfer_reward_type=reward_type)


class _TrainVisualizer:
  def __init__(self,
               trainer: "imitation.algorithms.adversarial.AdversarialTrainer",
               show_plots: bool,
               n_episodes_per_reward_data: int,
               log_dir: str,
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
    self.log_dir = log_dir
    self.expert_policy = expert_policy
    self.plot_idx = 0
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
    self._savefig("plot_fight_loss_disc", self.show_plots)

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
      self._savefig("plot_fight_epreward_gen", self.show_plots)

  def _savefig(self, prefix="", also_show=True):
    path = osp.join(self.log_dir, f"{prefix}_{self.plot_idx}")
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
