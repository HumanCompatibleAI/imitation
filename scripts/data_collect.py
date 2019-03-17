import logging

import stable_baselines

import yairl.util as util
import gin.tf
import argparse


def make_PPO2(env_name):
    """
    Hyperparameters and a vectorized environment for training a PPO2 expert.
    """
    env = util.make_vec_env(env_name, 8)
    # Didn't look at rl-baselines-zoo for this, but these hyperparameters
    # seem ok. They aren't great though.
    policy = stable_baselines.PPO2(util.FeedForward32Policy, env,
            verbose=0, tensorboard_log="output/",
            learning_rate=3e-3,
            nminibatches=32,
            noptepochs=10,
            n_steps=2048)
    return policy


@gin.configurable
def main(env_name, total_timesteps):
    logging.getLogger().setLevel(logging.INFO)

    policy = make_PPO2(env_name)

    callback = util.make_save_policy_callback("data/", env_name, 1)
    policy.learn(total_timesteps, callback=callback)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin_config", default='configs/cartpole_data_collect.gin')
    args = parser.parse_args()

    gin.parse_config_file(args.gin_config)

    main()
