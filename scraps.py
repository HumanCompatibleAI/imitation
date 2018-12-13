"""
MVP-not-really
"""

import os.path
import numpy as np
import gym
import tensorflow as tf

import stable_baselines
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

import reward_net
import util


def make_blank_policy(env, policy_network_class=MlpPolicy,
        init_tensorboard=True, policy_class=stable_baselines.PPO1):
    """
    Instantiates a policy for the provided environment.

    Params:
    env (str or Env): The Env or its string id in Gym.
    policy_network_class (stable_baselines.BasePolicy): A policy network
      constructor from the stable_baselines module.
    policy_class (stable_baselines.BaseRLModel subclass): A policy constructor
      from the stable_baselines module.
    init_tensorboard (bool): If True, then initialize the policy to make
      TensorBoard summary writes.

    Return:
    policy (stable_baselines.BaseRLModel)
    """
    env = util.maybe_load_env(env)
    policy = policy_class(policy_network_class, env, verbose=1,
            optim_stepsize=0.0005,
            tensorboard_log="./output/{}/".format(env.spec.id))
    return policy


def get_trained_policy(env, force_train=False, timesteps=500000,
        never_overwrite=False, policy_class=stable_baselines.PPO1):
    """
    Returns a trained policy, maybe pretrained.

    If a policy for the environment hasn't been trained and pickled before,
    then first train and pickle it. Otherwise, load that pickled policy.

    Params:
    env (str or Env): The Env that this policy is meant to act in, or the
      string name of the Gym environment.
    timesteps (int): The number of training timesteps.
    force_train (bool): If True, then always train and pickle first, even
      if the policy already exists.
    never_overwrite (bool): It True, then don't pickle a policy if it means
      overwriting another pickle. Ah, pickles.
    policy_class (stable_baselines.BaseRLModel class): A policy constructor
      from the stable_baselines module.

    Return:
    policy (stable_baselines.BaseRLModel)
    """
    env = util.maybe_load_env(env)
    savepath = "saved_models/{}_{}.pkl".format(
            policy_class.__name__, env.spec.id)
    exists = os.path.exists(savepath)

    if exists and not force_train:
        policy = policy_class.load(savepath, env=env)
        print("loaded policy from '{}'".format(savepath))
    else:
        print("Didn't find pickled policy at {}. Training...".format(savepath))
        policy = make_blank_policy(env, policy_class=policy_class)
        policy.learn(timesteps)
        if exists and never_overwrite:
            print(("Avoided saving policy pickle at {} because overwrite "
                    "is disabled and that file already exists!"
                    ).format(savepath))
        else:
            policy.save(savepath)
            print("Saved pickle!")
    return policy


def generate_rollouts(policy, n_timesteps, env=None):
    """
    Generate state-action pairs from a policy.

    Params:
    policy (stable_baselines.BaseRLModel) -- A stable_baselines Model, trained
      on the gym environment.
    env (Env or str or None) -- A Gym Env. VecEnv is not currently supported.
      If env is None, then use policy.env.
    n_timesteps (int) -- The number of state-action pairs to collect.

    Return:
    rollout_obs (array) -- A numpy array with shape
      `[n_timesteps] + env.observation_space.shape`.
    rollout_act (array) -- A numpy array with shape
      `[n_timesteps] + env.action_space.shape`.
    """
    if env is None:
        env = policy.env
    else:
        env = util.maybe_load_env(env)
        policy.set_env(env)  # This checks that env and policy are compatbile.

    rollout_obs = []
    rollout_act = []
    while len(rollout_obs) < n_timesteps:
        done = False
        obs = env.reset()
        while not done and len(rollout_obs) < n_timesteps:
            act, _ = policy.predict(obs)
            rollout_obs.append(obs)
            rollout_act.append(act)
            obs, _, done, _ = env.step(act)

    rollout_obs = np.array(rollout_obs)
    exp_obs = (n_timesteps,) + env.observation_space.shape
    assert rollout_obs.shape == exp_obs

    rollout_act = np.array(rollout_act)
    exp_act = (n_timesteps,) + env.action_space.shape
    assert rollout_act.shape == exp_act

    return rollout_obs, rollout_act


def discriminate_rollouts(env="CartPole-v1"):
    """
    1. Generate rollouts from expert policy.
    2. Generate random rollouts.
    3. Train a Discriminator to distinguish betwen expert and random s-a pairs.
    4. Show loss decreasing over time.
    5. Show accuracy over training set. (Don't care about validation for now.)
    """
    expert_policy = get_trained_policy(env)
    fake_policy = make_blank_policy(env)
    rnet = reward_net.BasicRewardNet(env)
    ## DEBUG
    env = expert_policy.env
    obv = env.reset()
    pol = expert_policy
    expert_rollout_obs, expert_rollout_act = generate_rollouts(pol, 100)
    act_prob = util.action_prob_rollout(pol, expert_rollout_obs,
            expert_rollout_act)


discriminate_rollouts('CartPole-v1')
