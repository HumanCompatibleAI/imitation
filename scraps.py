"""
MVP-not-really
"""
import gym
import numpy as np
import tensorflow as tf

import stable_baselines
import reward_net
import util


def discriminate_rollouts(env="CartPole-v1"):
    """
    1. Generate rollouts from expert policy.
    2. Generate random rollouts.
    3. Train a Discriminator to distinguish betwen expert and random s-a pairs.
    4. Show loss decreasing over time.
    5. Show accuracy over training set. (Don't care about validation for now.)
    """
    env = util.maybe_load_env(env, vectorize=True)
    expert_policy = util.get_trained_policy(env)
    fake_policy = util.make_blank_policy(env)
    rnet = reward_net.BasicRewardNet(env)
    ## DEBUG
    obs = env.reset()
    pol = expert_policy
    util.reset_and_wrap_env_reward(env, lambda s, a, s_p: 3)
    expert_obs_old, expert_act, expert_obs_new = util.generate_rollouts(
            pol, env, 1000)
    act_prob = util.rollout_action_probability(pol, expert_obs_old,
            expert_act)

discriminate_rollouts('CartPole-v1')
