import logging

import stable_baselines

from scripts import data_load_experts
from yairl.airl import AIRLTrainer
from yairl.reward_net import BasicRewardNet
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
    return env, policy

def main():
    logging.getLogger().setLevel(logging.INFO)
    experts = data_load_experts(savedir="data", file_prefix="pendulum",
            policy_class=stable_baselines.PPO2,
            policy_load_opt=dict(policy=util.FeedForward32Policy),
            n_experts=5)

    env, gen_policy = make_PPO2()
    rn = BasicRewardNet(env)
    trainer = AIRLTrainer(env, gen_policy, rn, expert_policies=experts)
    trainer.train()

if __name__ == "__main__":
    main()
