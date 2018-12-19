"""
Random experiments. As this file expands, I will probably move individual
experiments into an scripts/ directory.
"""
import datetime

from matplotlib import pyplot as plt
import tqdm

from airl import AIRLTrainer
from reward_net import BasicRewardNet
import util


def _init_trainer(env, use_expert_rollouts=True):
    """
    Initialize an AIRL trainer to train a BasicRewardNet (discriminator)
    versus a policy (generator).

    Params:
    use_expert_rollouts (bool) -- If True, then load an expert policy to
      generate training data, and error if this expert policy doesn't exist for
      this environment. If False, then generate random rollouts.

    Return:
    trainer (AIRLTrainer) -- The AIRL trainer.
    """
    if isinstance(env, str):
        env = util.make_vec_env(env, 32)
    else:
        env = util.maybe_load_env(env, True)
    policy = util.make_blank_policy(env, init_tensorboard=False)
    if use_expert_rollouts:
        rollout_policy = util.load_expert_policy(env)
        if rollout_policy is None:
            raise ValueError(env)
    else:
        rollout_policy = policy

    obs_old, act, obs_new, _ = util.rollout_generate(rollout_policy, env,
            n_timesteps=1000)

    rn = BasicRewardNet(env)
    trainer = AIRLTrainer(env, policy=policy, reward_net=rn,
            expert_obs_old=obs_old, expert_act=act, expert_obs_new=obs_new)
    return trainer


def plot_episode_reward_vs_time(env='CartPole-v1', n_episodes=50,
        n_epochs_per_plot=250, n_plots=100):
    """
    Make sure that generator policy trained to mimick expert policy
    demonstrations) achieves higher reward than a random policy.

    In other words, perform a basic check on the imitation learning
    capabilities of AIRLTrainer.
    """
    trainer = _init_trainer(env, use_expert_rollouts=True)
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
    plt.savefig("output/plot_episode_reward_vs_time_{}.png".format(
        datetime.datetime.now()
        ))


def plot_discriminator_loss(env='CartPole-v1', n_steps_per_plot=1000,
        n_plots=100, n_gen_warmup_steps=500):
    """
    Train the generator briefly, and then

    Train the discriminator to distinguish (unchanging) expert rollouts versus
    the unchanging random rollouts for a long time and plot discriminator loss.
    """
    trainer = _init_trainer(env, use_expert_rollouts=True)
    n_timesteps = len(trainer.expert_obs_old)
    (gen_obs_old, gen_act, gen_obs_new, _) = util.rollout_generate(
            trainer.policy, trainer.env, n_timesteps=n_timesteps)
    kwargs = dict(gen_obs_old=gen_obs_old, gen_act=gen_act,
            gen_obs_new=gen_obs_new)
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
    for _ in tqdm.trange(n_plots, desc="discriminator"):
        epoch()
        add_plot()

    plt.plot(X, Y, label="discriminator loss")
    plt.legend()
    plt.savefig("output/plot_discriminator_loss_{}.png".format(
        datetime.datetime.now()
        ))


def plot_generator_loss(env='CartPole-v1', n_steps_per_plot=5000,
        n_plots=100, n_disc_warmup_steps=100):
    """
    Train the discriminator briefly, and then

    Train the generator to distinguish (unchanging) expert rollouts to
    confuse the discriminator, and plot discriminator loss.
    """
    trainer = _init_trainer(env, use_expert_rollouts=True)
    n_timesteps = len(trainer.expert_obs_old)

    (gen_obs_old, gen_act, gen_obs_new, _) = util.rollout_generate(
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
    for _ in tqdm.trange(n_plots, desc="generator"):
        epoch()
        add_plot()

    plt.plot(X, Y, label="discriminator loss")
    plt.legend()
    plt.savefig("output/plot_generator_loss_{}.png".format(
        datetime.datetime.now()
        ))


def plot_fight_loss(env='CartPole-v1',
        n_epochs=100,
        n_plots_each_per_epoch=50,
        n_disc_steps_per_plot=100,
        n_gen_steps_per_plot=10000):
    """
    Alternate between training the generator and discriminator.

    Plot discriminator loss during discriminator training steps in blue and
    discriminator loss during generator training steps in red.
    """
    trainer = _init_trainer(env, use_expert_rollouts=True)
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
    def add_plot(gen_mode=False):
        mode = "gen" if gen_mode else "dis"
        X, Y = gen_data if gen_mode else disc_data
        X.append(plot_idx)
        Y.append(trainer.eval_disc_loss())
        print("plot idx ({}): {}".format(mode, plot_idx))
        print("disc loss: {}".format(Y[-1]))

    add_plot(False)
    for _ in tqdm.trange(n_epochs, desc="epoch"):
        for _ in range(n_plots_each_per_epoch):
            epoch(False)
            add_plot(False)
        for _ in range(n_plots_each_per_epoch):
            epoch(True)
            add_plot(True)

    plt.scatter(disc_data[0], disc_data[1], c='g',
            label="discriminator loss (dis step)")
    plt.scatter(gen_data[0], gen_data[1], c='r',
            label="discriminator loss (gen step)")
    plt.legend()
    plt.savefig("output/plot_fight_loss_{}.png".format(
        datetime.datetime.now()
        ))


if __name__ == "__main__":
    plot_fight_loss()
