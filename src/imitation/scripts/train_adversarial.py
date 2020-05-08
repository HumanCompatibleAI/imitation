"""Train GAIL or AIRL and plot its output.

Can be used as a CLI script, or the `train_and_plot` function can be called directly.
"""

import os
import os.path as osp
from typing import Optional

import tensorflow as tf
from sacred.observers import FileStorageObserver

from imitation.algorithms.adversarial import init_trainer
from imitation.data import rollout, types
from imitation.policies import serialize
from imitation.scripts.config.train_adversarial import train_ex
from imitation.util import networks
from imitation.util import sacred as sacred_util


def save(trainer, save_path):
    """Save discriminator and generator."""
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    trainer.discrim.save(os.path.join(save_path, "discrim"))
    # TODO(gleave): unify this with the saving logic in data_collect?
    # (Needs #43 to be merged before attempting.)
    serialize.save_stable_model(
        os.path.join(save_path, "gen_policy"),
        trainer.gen_policy,
        trainer.venv_train_norm,
    )


@train_ex.main
def train(
    _run,
    _seed: int,
    env_name: str,
    rollout_path: str,
    n_expert_demos: Optional[int],
    log_dir: str,
    init_trainer_kwargs: dict,
    total_timesteps: int,
    n_episodes_eval: int,
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
        `checkpoint_interval` epochs and after training is complete. If 0,
        then only save weights after training is complete. If <0, then don't
        save weights at all.

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
    expert_trajs = types.load(rollout_path)

    if n_expert_demos is not None:
        assert len(expert_trajs) >= n_expert_demos
        expert_trajs = expert_trajs[:n_expert_demos]

    expert_stats = rollout.rollout_stats(expert_trajs)

    with networks.make_session():
        if init_tensorboard:
            sb_tensorboard_dir = osp.join(log_dir, "sb_tb")
            kwargs = init_trainer_kwargs
            kwargs["init_rl_kwargs"] = kwargs.get("init_rl_kwargs", {})
            kwargs["init_rl_kwargs"]["tensorboard_log"] = sb_tensorboard_dir

        trainer = init_trainer(
            env_name, expert_trajs, seed=_seed, log_dir=log_dir, **init_trainer_kwargs
        )

        def callback(epoch):
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                save(trainer, os.path.join(log_dir, "checkpoints", f"{epoch:05d}"))

        trainer.train(total_timesteps, callback)

        # Save final artifacts.
        if checkpoint_interval >= 0:
            save(trainer, os.path.join(log_dir, "checkpoints", "final"))

        # Final evaluation of imitation policy.
        results = {}
        sample_until_eval = rollout.min_episodes(n_episodes_eval)
        trajs = rollout.generate_trajectories(
            trainer.gen_policy, trainer.venv_test, sample_until=sample_until_eval
        )
        results["imit_stats"] = rollout.rollout_stats(trajs)
        results["expert_stats"] = expert_stats
        return results


def main_console():
    observer = FileStorageObserver.create(osp.join("output", "sacred", "train"))
    train_ex.observers.append(observer)
    train_ex.run_commandline()


if __name__ == "__main__":
    main_console()
