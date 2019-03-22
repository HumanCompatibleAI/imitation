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
import tqdm

import yairl.util as util
from yairl.util.trainer import init_trainer


# TODO: This is cruft. It was mostly useful for prototyping, but
# I should get rid of it because it doesn't really add anything.
def data_train_and_save_experts(policy, *, total_timesteps, savedir,
        file_prefix, save_interval=1, policy_learn_opt=None):
    """
    Train an policy and save the number of environment

    Params:
    policy (stable_baselines.BaseRLModel): The policy to train.
    total_timesteps (int): The total_timesteps argument for policy.learn(). In
      other words, the number of timesteps to train for.
    savedir (str) -- The directory to save pickle files to.
    file_prefix (str) -- A prefix for the pickle file.
    save_interval (int): The number of training timesteps in between saves.
    policy_learn_opt (dict): Additional keyword arguments to policy.learn().
    """
    policy_learn_opt = policy_learn_opt or {}
    callback = util.make_save_policy_callback(savedir, file_prefix,
            save_interval)
    policy.learn(total_timesteps, callback=callback, **policy_learn_opt)


def data_load_experts(*, savedir, file_prefix, policy_class, n_experts,
        policy_load_opt={}):
    """
    Load expert policies saved using data_collect.

    Params:
    savedir (str) -- The directory containing pickle files, same as in
      `data_collect()`.
    file_prefix (str) -- The prefix for pickle filenames, same as in
      `data_collect()`.
    n_experts (int) -- The number of experts to load. We prioritize recent
      iterations over earlier iterations.
    policy_class (stable_baselines.BaseRLModel class) -- The class of the
      pickled policy.
    policy_load_opt (dict) -- Keyword arguments for `policy_class.load()`.
      Must set policy=CustomPolicy if using a custom policy class.
    """
    assert n_experts > 0

    def ith_file(i):
        return os.path.join(savedir, "{}-{}.pkl".format(file_prefix, i))

    # XXX: Use a number-aware sorted glob instead of a linear search.
    # We could get a sorted list and simply take the last n_experts elements.
    n = 1
    while os.path.exists(ith_file(n)):
        n += 1

    if n - 1 < n_experts:
        raise ValueError(
            """
            Wanted to load {} experts, but there were only {} experts at
            {}-*.pkl
            """.format(n_experts, n - 1))

    policy_load_opt = policy_load_opt or {}
    expert_pols = []
    for i in range(n - n_experts, n):
        logging.info("Loading expert {}".format(i))
        pol = policy_class.load(ith_file(i), **policy_load_opt)
        expert_pols.append(pol)
    return expert_pols


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
        logging.info("Epoch {}".format(trainer.epochs_so_far))
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


def plot_fight_loss(env='CartPole-v1',
        n_epochs=70,
        n_plots_each_per_epoch=10,
        n_disc_steps_per_plot=10,
        n_gen_steps_per_plot=10000,
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
    trainer = trainer or init_trainer(env, n_expert_timesteps=n_rollout_samples)
    if trainer_hook_fn:
        trainer_hook_fn(trainer)

    os.makedirs("output/", exist_ok=True)

    plot_idx = 0
    def epoch(gen_mode=False):
        nonlocal plot_idx
        if gen_mode:
            trainer.train_gen(n_steps=n_gen_steps_per_plot)
        else:
            trainer.train_disc(n_steps=n_disc_steps_per_plot)
        plot_idx += 1

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
        rand_policy = util.make_blank_policy(env_vec)

        gen_rew = util.rollout.total_reward(gen_policy, env_vec,
                                            n_episodes=n_gen_plot_episodes) / n_gen_plot_episodes
        rand_rew = util.rollout.total_reward(rand_policy, env_vec,
                                             n_episodes=n_gen_plot_episodes) / n_gen_plot_episodes
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
    for epoch_num in tqdm.trange(n_epochs, desc="epoch"):
        for _ in range(n_plots_each_per_epoch):
            epoch(False)
            add_plot_disc(False)
        for _ in range(n_plots_each_per_epoch):
            epoch(True)
            add_plot_disc(True)
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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    plot_fight_loss(interactive=False)
