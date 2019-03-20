"""
Random experiments. As this file expands, I will probably move individual
experiments into an scripts/ directory.

These scripts are meant to be run in a Jupyter notebook (displays figures)
but also automatically save timestamped figures to the output/ directory.
"""
import datetime
import logging
import os

from matplotlib import pyplot as plt
import tensorflow as tf
import tqdm

from yairl.util.trainer import init_trainer
import yairl.util as util
import gin.tf


def plot_episode_reward_vs_time(env='CartPole-v1', n_episodes=50,
        n_epochs_per_plot=250, n_plots=100):
    """
    Make sure that generator policy trained to mimick expert policy
    demonstrations) achieves higher reward than a random policy.

    In other words, perform a basic check on the imitation learning
    capabilities of AIRLTrainer.
    """
    trainer = init_trainer(env)
    expert_policy = util.load_expert_policy(env)
    random_policy = util.make_blank_policy(env)
    gen_policy = trainer.policy

    assert expert_policy is not None
    assert random_policy is not None

    X, random_rews, gen_rews = [], [], []

    def add_single_data(policy, policy_name, lst):
        rew = util.rollout.total_reward(policy, env, n_episodes=n_episodes)
        lst.append(rew)
        logging.info("{} reward:".format(policy_name), rew)

    def make_data():
        X.append(trainer.epochs_so_far)
        loggin.info("Epoch {}".format(trainer.epochs_so_far))
        # add_single_data(expert_policy, "expert", expert_rews)
        add_single_data(random_policy, "random", random_rews)
        add_single_data(gen_policy, "generator", gen_rews)

    make_data()
    for _ in range(n_plots):
        trainer.train(n_epochs=n_epochs_per_plot)
        make_data()

    # plt.plot(X, expert_rews, label="expert")
    plt.plot(X, gen_rews, label="generator")
    plt.plot(X, random_rews, label="random")
    plt.legend()
    _savefig_timestamp("plot_episode_reward_vs_time")

@gin.configurable
def plot_fight_loss(policy_dir, env='CartPole-v1',
        n_epochs=70,
        n_plots_each_per_epoch=10,
        n_disc_steps_per_epoch=100,
        n_gen_steps_per_epoch=100000,
        n_rollout_samples=4000,
        n_gen_plot_episodes=10,
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
    trainer = trainer or init_trainer(env, policy_dir=policy_dir, n_expert_timesteps=n_rollout_samples)
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
        mode = "gen" if gen_mode else "dis"
        X, Y = gen_data if gen_mode else disc_data
        X.append(plot_idx)
        Y.append(trainer.eval_disc_loss())
        logging.info("plot idx ({}): {} disc loss: {}"
                .format(mode, plot_idx, Y[-1]))
    def show_plot_disc():
        plt.scatter(disc_data[0], disc_data[1], c='g', alpha=0.7, s=4,
                label="discriminator loss (dis step)")
        plt.scatter(gen_data[0], gen_data[1], c='r', alpha=0.7, s=4,
                label="discriminator loss (gen step)")
        plt.title("epoch={}".format(epoch_num))
        plt.legend()
        _savefig_timestamp("plot_fight_loss_disc", interactive)

    gen_ep_reward = []
    rand_ep_reward = []
    def add_plot_gen():
        env_vec = util.make_vec_env(env, 8)
        gen_policy = trainer.gen_policy
        rand_policy = util.make_blank_policy(env)

        gen_rew = util.rollout.total_reward(gen_policy, env,
                n_episodes=n_gen_plot_episodes)/n_gen_plot_episodes
        rand_rew = util.rollout.total_reward(rand_policy, env,
                n_episodes=n_gen_plot_episodes)/n_gen_plot_episodes
        gen_ep_reward.append(gen_rew)
        rand_ep_reward.append(rand_rew)
        logging.info("generator reward: {}".format(gen_rew))
        logging.info("random reward: {}".format(rand_rew))
    def show_plot_gen():
        plt.title("Cartpole performance (expert=500)")
        plt.xlabel("epochs")
        plt.ylabel("Average reward per episode (n={})"
                .format(n_gen_plot_episodes))
        plt.plot(gen_ep_reward, label="avg gen ep reward", c="red")
        plt.plot(rand_ep_reward, label="avg random ep reward", c="black")
        plt.legend()
        _savefig_timestamp("plot_fight_epreward_gen", interactive)

    add_plot_disc(False)
    add_plot_gen()

    n_gen_steps_per_plot = int(n_gen_steps_per_epoch / n_plots_each_per_epoch)
    n_disc_steps_per_plot = int(n_disc_steps_per_epoch / n_plots_each_per_epoch)

    def train_plot(steps, gen_mode):
        nonlocal plot_idx
        while steps > 0:
            steps_to_train = min(steps, n_gen_steps_per_plot)
            if gen_mode:
                trainer.train_gen(n_steps=n_gen_steps_per_plot)
            else:
                trainer.train_disc(n_steps=n_disc_steps_per_plot)
            steps -= steps_to_train
            add_plot_disc(gen_mode)
            plot_idx += 1

    for epoch_num in tqdm.trange(n_epochs, desc="epoch"):
        train_plot(n_gen_steps_per_epoch, True)
        train_plot(n_gen_steps_per_epoch, False)

        add_plot_gen()

        show_plot_disc()
        show_plot_gen()
        if trainer_hook_fn:
            trainer_hook_fn(trainer)

    return trainer, gen_data, disc_data, gen_ep_reward


def _savefig_timestamp(prefix="", also_show=True):
    path = "output/{}_{}.png".format(prefix, datetime.datetime.now())
    plt.savefig(path)
    logging.info("plot saved to {}".format(path))
    if also_show:
        plt.show()

def _decor_tf_init(f):
    with tf.Session() as sess:
        pass

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    plot_fight_loss(interactive=False)
