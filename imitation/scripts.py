"""
Random experiments. As this file expands, I will probably move individual
experiments into an scripts/ directory.

These scripts are meant to be run in a Jupyter notebook (displays figures)
but also automatically save timestamped figures to the output/ directory.
"""
from collections import defaultdict
import datetime
import os

import gin.tf
from matplotlib import pyplot as plt
import tensorflow as tf
import tqdm

import imitation.util as util
from imitation.util.trainer import init_trainer


@gin.configurable
def train_and_plot(policy_dir, env='CartPole-v1',
                   n_epochs=100,
                   n_plots_each_per_epoch=0,
                   n_disc_steps_per_epoch=10,
                   n_gen_steps_per_epoch=10000,
                   n_expert_timesteps=4000,
                   n_gen_plot_episodes=0,
                   trainer_hook_fn=None,
                   trainer=None,
                   interactive=True,
                   ):
    """
    Alternate between training the generator and discriminator.

    Every epoch:
    - Plot discriminator loss during discriminator training steps in blue and
    discriminator loss during generator training steps in red.
    - Plot the performance of the generator policy versus the performance of
      a random policy.
    """
    if trainer is None:
        trainer = init_trainer(
            env, policy_dir=policy_dir,
            n_expert_timesteps=n_expert_timesteps)
    if trainer_hook_fn:
        trainer_hook_fn(trainer)

    os.makedirs("output/", exist_ok=True)

    plot_idx = 0

    gen_data = ([], [])
    disc_data = ([], [])

    def add_plot_disc(gen_mode=False):
        """
        gen_mode (bool): Whether the generator or the discriminator is active.
          We use this to color the data points.
        """
        if n_plots_each_per_epoch <= 0:
            return

        mode = "gen" if gen_mode else "dis"
        X, Y = gen_data if gen_mode else disc_data
        X.append(plot_idx)
        Y.append(trainer.eval_disc_loss())
        tf.logging.info(
            "plot idx ({}): {} disc loss: {}"
            .format(mode, plot_idx, Y[-1]))

    def show_plot_disc():
        if n_plots_each_per_epoch <= 0:
            return

        plt.scatter(disc_data[0], disc_data[1], c='g', alpha=0.7, s=4,
                    label="discriminator loss (dis step)")
        plt.scatter(gen_data[0], gen_data[1], c='r', alpha=0.7, s=4,
                    label="discriminator loss (gen step)")
        plt.title("epoch={}".format(epoch_num))
        plt.legend()
        _savefig_timestamp("plot_fight_loss_disc", interactive)

    gen_ep_reward = defaultdict(list)
    rand_ep_reward = defaultdict(list)
    exp_ep_reward = defaultdict(list)

    def add_plot_gen(env, name):
        if n_gen_plot_episodes <= 0:
            return

        gen_policy = trainer.gen_policy
        rand_policy = util.make_blank_policy(env)
        exp_policy = trainer.expert_policies[-1]

        gen_rew = util.rollout.total_reward(
            gen_policy, env, n_episodes=n_gen_plot_episodes
        ) / n_gen_plot_episodes
        rand_rew = util.rollout.total_reward(
            rand_policy, env, n_episodes=n_gen_plot_episodes
        ) / n_gen_plot_episodes
        exp_rew = util.rollout.total_reward(
            exp_policy, env, n_episodes=n_gen_plot_episodes
        ) / n_gen_plot_episodes
        gen_ep_reward[name].append(gen_rew)
        rand_ep_reward[name].append(rand_rew)
        exp_ep_reward[name].append(exp_rew)
        tf.logging.info("generator reward: {}".format(gen_rew))
        tf.logging.info("random reward: {}".format(rand_rew))
        tf.logging.info("exp reward: {}".format(exp_rew))

    def show_plot_gen():
        if n_gen_plot_episodes <= 0:
            return

        for name in gen_ep_reward:
            plt.title(name + " Performance")
            plt.xlabel("epochs")
            plt.ylabel("Average reward per episode (n={})"
                       .format(n_gen_plot_episodes))
            plt.plot(gen_ep_reward[name], label="avg gen ep reward", c="red")
            plt.plot(rand_ep_reward[name],
                     label="avg random ep reward", c="black")
            plt.plot(exp_ep_reward[name], label="avg exp ep reward", c="blue")
            plt.legend()
            _savefig_timestamp("plot_fight_epreward_gen", interactive)

    add_plot_disc(False)
    add_plot_gen(env, "True Reward")
    add_plot_gen(trainer.wrap_env_test_reward(env), "Learned Reward")

    if n_plots_each_per_epoch <= 0:
        n_gen_steps_per_plot = float('Inf')
        n_disc_steps_per_plot = float('Inf')
    else:
        n_gen_steps_per_plot = int(round(
                n_gen_steps_per_epoch / n_plots_each_per_epoch))
        n_disc_steps_per_plot = int(round(
                n_disc_steps_per_epoch / n_plots_each_per_epoch))

    def train_plot_itr(steps, gen_mode, steps_per_plot):
        nonlocal plot_idx
        while steps > 0:
            steps_to_train = min(steps, steps_per_plot)
            if gen_mode:
                trainer.train_gen(n_steps=steps_to_train)
            else:
                trainer.train_disc(n_steps=steps_to_train)
            steps -= steps_to_train
            add_plot_disc(gen_mode)
            plot_idx += 1

    for epoch_num in tqdm.trange(n_epochs, desc="epoch"):
        train_plot_itr(n_disc_steps_per_epoch, False, n_disc_steps_per_plot)
        train_plot_itr(n_gen_steps_per_epoch, True, n_gen_steps_per_plot)

        add_plot_gen(env, "True Reward")
        add_plot_gen(trainer.wrap_env_test_reward(env), "Learned Reward")

        show_plot_disc()
        show_plot_gen()
        if trainer_hook_fn:
            trainer_hook_fn(trainer)

    return trainer, gen_data, disc_data, gen_ep_reward


def _savefig_timestamp(prefix="", also_show=True):
    path = "output/{}_{}.png".format(prefix, datetime.datetime.now())
    plt.savefig(path)
    tf.logging.info("plot saved to {}".format(path))
    if also_show:
        plt.show()
