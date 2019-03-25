import argparse

import gin.tf
import imitation.util as util
import stable_baselines
import tensorflow as tf
import numpy as np
from stable_baselines.common.policies import MlpPolicy



def make_PPO2(env_name):
    """
    Hyperparameters and a vectorized environment for training a PPO2 expert.
    """
    env = util.make_vec_env(env_name, 8)
    # Didn't look at rl-baselines-zoo for this, but these hyperparameters
    # seem ok. They aren't great though.
    policy = stable_baselines.PPO2(MlpPolicy, env,
                                   verbose=1, tensorboard_log="output/",
                                   learning_rate=3e-3,
                                   nminibatches=32,
                                   noptepochs=10,
                                   n_steps=2048)
    return policy, env


@gin.configurable
def main(env_name, total_timesteps):
    tf.logging.set_verbosity(tf.logging.INFO)

    policy, env = make_PPO2(env_name)

    callback = util.make_save_policy_callback("data/")
    policy.learn(total_timesteps, callback=callback)

    # total_reward = 0
    # # Enjoy trained agent
    # obs = env.reset()
    # done = [False]
    # while not all(done):
    #     action, _states = policy.predict(obs)
    #     obs, reward, done, info = env.step(action)
    #     total_reward += reward
    # print("total_reward", np.mean(total_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--gin_config", default='configs/cartpole_data_collect.gin')
    args = parser.parse_args()

    gin.parse_config_file(args.gin_config)

    main()
