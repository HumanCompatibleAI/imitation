import logging

import stable_baselines

from scripts import data_train_and_save_experts
import yairl.util as util

def make_PPO2():
    """
    Hyperparameters and a vectorized environment for training a PPO2 expert.
    """
    env = util.make_vec_env("Pendulum-v0", 8)
    policy = stable_baselines.PPO2(util.FeedForward32Policy, env,
            verbose=0, tensorboard_log="output/",
            learning_rate=3e-3,
            nminibatches=32,
            noptepochs=10,
            n_steps=2048)
    return policy

def make_TRPO():
    """
    Hyperparameters and a nonvectorized environment for training a TRPO
    expert. (TRPO doesn't support vectorized environments).

    Warning -- These do not successfully train Pendulum! For now use
      PPO instead.
    """
    policy = stable_baselines.TRPO(util.FeedForward32Policy, "Pendulum-v0",
            verbose=0, tensorboard_log="output/",
            # Maybe try fiddling with vf_stepsize. The other hyperparameters
            # I think match the obvious hyperparameters in justinfu/inverse-rl.
            vf_stepsize=2e-3,
            vf_iters=200)
    return policy

def main():
    logging.getLogger().setLevel(logging.INFO)

    # policy = util.make_blank_policy(env, policy_class=stable_baselines.TRPO)
    policy = make_PPO2()
    data_train_and_save_experts(policy, total_timesteps=600000,
            savedir="data/", file_prefix="pendulum")


if __name__ == "__main__":
    main()
