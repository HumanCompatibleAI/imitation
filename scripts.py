"""
Random experiments. As this file expands, I will probably move individual
experiments into an scripts/ directory.

These scripts are meant to be run in a Jupyter notebook (displays figures)
but also automatically save timestamped figures to the output/ directory.
"""
import datetime
import os

from matplotlib import pyplot as plt
import tensorflow as tf
import tqdm

from yairl.trainer_util import init_trainer
import yairl.util as util


def data_train_and_save_experts(policy, env, *, total_timesteps, savedir,
        file_prefix, save_interval=250, policy_learn_opt=None):
    """
    Train an policy and save the number of environment

    Params:
    policy (stable_baselines.BaseRLModel): The policy to train.
    env (gym.Env): The environment to train.
    total_timesteps (int): The total_timesteps argument for policy.learn(). In
      other words, the number of timesteps to train for.
    savedir (str) -- The directory to save pickle files to.
    file_prefix (str) -- A prefix for the pickle file.
    save_interval (int): The number of training timesteps in between saves.
    policy_learn_opt (dict): Additional keyword arguments to policy.learn().
    """
    callback_opt = callback_opt or {}
    policy_learn_opt = policy_learn_opt or {}
    callback = util.make_save_policy_callback(savedir, save_prefix,
            **callback_opt)
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
      (See https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html)
    """
    assert n_experts > 0

    # XXX: Use a number-aware sorted glob instead of a linear search.
    # We could get a sorted list and simply take the last n_experts elements.
    n = 1
    while os.path.join(savedir, "{}-{}.pkl".format(file_prefix, n)).exists():
        n += 1

    if n - 1 < n_experts:
        raise ValueError(
            """
            Wanted to load {} experts, but there were only {} experts at
            {}-*.pkl
            """.format(n_experts, i - 1))

    policy_load_opt = policy_load_opt or {}
    expert_pols = []
    for i in range(n - n_experts, n):
        pol = policy_class.load("{}-{}.pkl".format(file_prefix, i), **kwargs)
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
        rew = util.rollout_total_reward(policy, env, n_episodes=n_episodes)
        lst.append(rew)
        print("{} reward:".format(policy_name), rew)

    def make_data():
        X.append(trainer.epochs_so_far)
        print("Epoch {}".format(trainer.epochs_so_far))
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


def plot_discriminator_loss(env='CartPole-v1', n_steps_per_plot=1000,
        n_plots=100, n_gen_warmup_steps=500):
    """
    Train the generator briefly, and then

    Train the discriminator to distinguish (unchanging) expert rollouts versus
    the unchanging random rollouts for a long time and plot discriminator loss.
    """
    trainer = init_trainer(env)
    n_timesteps = len(trainer.expert_obs_old)
    (gen_old_obs, gen_act, gen_new_obs, _) = util.rollout_generate(
            trainer.policy, trainer.env, n_timesteps=n_timesteps)
    kwargs = dict(gen_old_obs=gen_old_obs, gen_act=gen_act,
            gen_new_obs=gen_new_obs)
    trainer.train_gen(n_steps=n_gen_warmup_steps)

    steps_so_far = 0
    def epoch():
        nonlocal steps_so_far
        trainer.train_disc(**kwargs, n_steps=n_steps_per_plot)
        steps_so_far += n_steps_per_plot

    X = []
    Y = []
    def add_plot():
        X.append(steps_so_far)
        Y.append(trainer.eval_disc_loss(**kwargs))
        print("step: {}".format(steps_so_far))
        print("loss: {}".format(Y[-1]))

    add_plot()
    for _ in tqdm.tnrange(n_plots, desc="discriminator"):
        epoch()
        add_plot()

    plt.plot(X, Y, label="discriminator loss")
    plt.legend()
    _savefig_timestamp("plot_discriminator_loss")


def plot_generator_loss(env='CartPole-v1', n_steps_per_plot=5000,
        n_plots=100, n_disc_warmup_steps=100):
    """
    Train the discriminator briefly, and then

    Train the generator to distinguish (unchanging) expert rollouts to
    confuse the discriminator, and plot discriminator loss.
    """
    trainer = init_trainer(env)
    n_timesteps = len(trainer.expert_obs_old)

    (gen_old_obs, gen_act, gen_new_obs, _) = util.rollout_generate(
            trainer.policy, trainer.env, n_timesteps=n_timesteps)

    steps_so_far = 0
    def epoch():
        nonlocal steps_so_far
        trainer.train_gen(n_steps=n_steps_per_plot)
        steps_so_far += n_steps_per_plot

    X = []
    Y = []
    def add_plot():
        X.append(steps_so_far)
        Y.append(trainer.eval_disc_loss())
        print("step: {}".format(steps_so_far))
        print("disc loss: {}".format(Y[-1]))

    add_plot()
    for _ in tqdm.tnrange(n_plots, desc="generator"):
        epoch()
        add_plot()

    plt.plot(X, Y, label="discriminator loss")
    plt.legend()
    _savefig_timestamp("plot_generator_loss")


def plot_fight_loss(env='CartPole-v1',
        n_epochs=100,
        n_plots_each_per_epoch=10,
        n_disc_steps_per_plot=500,
        n_gen_steps_per_plot=50000,
        n_rollout_samples=1000,
        n_gen_plot_episodes=100,
        trainer_hook_fn=None,
        trainer=None):
    """
    Alternate between training the generator and discriminator.

    Every epoch:
    - Plot discriminator loss during discriminator training steps in blue and
    discriminator loss during generator training steps in red.
    - Plot the performance of the generator policy versus the performance of
      a random policy.
    """
    trainer = trainer or init_trainer(env, n_expert_samples=n_rollout_samples)
    trainer_hook_fn(trainer)
    n_timesteps = len(trainer.expert_obs_old)

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
        print("plot idx ({}): {}".format(mode, plot_idx), end=" ")
        print("disc loss: {}".format(Y[-1]))
    def show_plot_disc():
        plt.scatter(disc_data[0], disc_data[1], c='g', alpha=0.7, s=4,
                label="discriminator loss (dis step)")
        plt.scatter(gen_data[0], gen_data[1], c='r', alpha=0.7, s=4,
                label="discriminator loss (gen step)")
        plt.title("epoch={}".format(epoch_num))
        plt.legend()
        _savefig_timestamp("plot_fight_loss_disc")

    gen_ep_reward = []
    rand_ep_reward = []
    def add_plot_gen():
        env_vec = util.make_vec_env(env, 8)
        gen_policy = trainer.policy
        rand_policy = util.make_blank_policy(env)

        gen_rew = util.rollout_total_reward(gen_policy, env,
                n_episodes=n_gen_plot_episodes)/n_gen_plot_episodes
        rand_rew = util.rollout_total_reward(rand_policy, env,
                n_episodes=n_gen_plot_episodes)/n_gen_plot_episodes
        gen_ep_reward.append(gen_rew)
        rand_ep_reward.append(rand_rew)
        print("generator reward:", gen_rew)
        print("random reward:", rand_rew)
    def show_plot_gen():
        plt.title("Cartpole performance (expert=500)")
        plt.xlabel("epochs")
        plt.ylabel("Average reward per episode (n={})"
                .format(n_gen_plot_episodes))
        plt.plot(gen_ep_reward, label="avg gen ep reward", c="red")
        plt.plot(rand_ep_reward, label="avg random ep reward", c="black")
        plt.legend()
        _savefig_timestamp("plot_fight_epreward_gen")

    add_plot_disc(False)
    add_plot_gen()
    for epoch_num in tqdm.tnrange(n_epochs, desc="epoch"):
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
    plt.show()

def _decor_tf_init(f):
    with tf.Session() as sess:
        pass

if __name__ == "__main__":
    plot_fight_loss()
