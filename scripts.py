"""
Random experiments. As this file expands, I will probably move individual
experiments into an scripts/ directory.
"""
import datetime

from matplotlib import pyplot as plt

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
        env = util.make_vec_env(env, 8)
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


def plot_discriminator_loss(env='CartPole-v1', n_epochs_per_plot=250,
        n_plots=100):
    """
    Train the discriminator to distinguish (unchanging) expert rollouts versus
    the unchanging random rollouts for a long time and plot discriminator loss.
    """

    trainer = _init_trainer(env, use_expert_rollouts=True)
    n_timesteps = len(self.trainer.expert_obs_old)

    (gen_obs_old, gen_act, gen_obs_new, _) = util.rollout_generate(
            self.trainer.policy, self.trainer.env, n_timesteps=n_timesteps)

    trainer.train_disc(self.trainer.expert_obs_old, self.trainer.expert_act,
            self.trainer.expert_obs_new, )


if __name__ == "__main__":
    plot_episode_reward_vs_time()
