import logging

import stable_baselines

from yairl.scripts import data_train_and_save_experts
import yairl.util as util

def make_PPO2():
    """
    Hyperparameters and a vectorized environment for training a PPO2 expert.
    """
    env = util.make_vec_env("CartPole-v1", 8)
    # Didn't look at rl-baselines-zoo for this, but these hyperparameters
    # seem ok. They aren't great though.
    policy = stable_baselines.PPO2(util.FeedForward32Policy, env,
            verbose=0, tensorboard_log="output/",
            learning_rate=3e-3,
            nminibatches=32,
            noptepochs=10,
            n_steps=2048)
    return policy

def main():
    logging.getLogger().setLevel(logging.INFO)

    policy = make_PPO2()
    data_train_and_save_experts(policy, total_timesteps=400000,
            savedir="data/", file_prefix="cartpole")


if __name__ == "__main__":
    main()
