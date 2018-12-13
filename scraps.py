"""
MVP-not-really
"""
import gym
import numpy as np
import tensorflow as tf

import stable_baselines
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

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
    expert_ro_obs, expert_ro_act = util.generate_rollouts(pol, env, 1000)
    act_prob = util.action_prob_rollout(pol, expert_ro_obs,
            expert_ro_act)
    util.reset_and_wrap_env_reward(env, lambda s, a, s_p: 3)
    expert_ro_obs, expert_ro_act = util.generate_rollouts(pol, env, 1000)
    act_prob = util.action_prob_rollout(pol, expert_ro_obs, expert_ro_act)

discriminate_rollouts('CartPole-v1')
