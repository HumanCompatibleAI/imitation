import gym
import numpy as np
import os.path

import stable_baselines
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecEnv

import reward_net
import util


def maybe_load_env(env_or_str, vectorize=False):
    """
    Params:
    env_or_str (str or gym.Env): The Env or its string id in Gym.
    vectorize (bool): If True, then vectorize the environment before returning,
      if it isn't already vectorized.

    Return:
    env (gym.Env) -- Either the original argument if it was an Env or an
      instantiated gym Env if it was a string.
    id (str) -- The environment's id.
    """
    if isinstance(env_or_str, str):
        env = gym.make(env_or_str)
    else:
        env = env_or_str

    if not is_vec_env(env) and vectorize:
        env = DummyVecEnv([lambda: env])

    return env


def is_vec_env(env):
    return isinstance(env, VecEnv)


def get_env_id(env):
    if is_vec_env(env):
        env = env.envs[0]
    return env.spec.id


def reset_and_wrap_env_reward(env, R):
    """
    Reset the VecEnv, and then wrap its step function so that it
    returns a custom reward based on state-action-new_state tuples.

    The old step function is saved as `env._orig_step_`.

    Param:
      env [gym.VecEnv] -- An vectorized Environment to modify in place.
      R [callable] -- The new reward function. Takes three arguments,
        `old_obs`, `action`, and `new_obs`. Returns the new reward.
        - `old_obs` is the vector of observations seen before taking the action.
        - `action` is vector of actions passed to env.step().
        - `new_obs` is the vector of observations seen after taking the action.
            This is same as the observation vector returned by env.step().
    """
    assert is_vec_env(env)
    # XXX: Look at gym wrapper class which can override step in a
    # more idiomatic way.
    old_obs = env.reset()

    orig = getattr(env, "_orig_step_", env.step)
    env._orig_step_ = orig
    def wrapped_step(action):
        nonlocal old_obs
        obs, reward, done, info = env._orig_step_(action)
        wrapped_reward = R(old_obs, action, obs)
        old_obs = obs
        return obs, wrapped_reward, done, info

    env.step = wrapped_step


def action_prob_rollout(policy, rollout_obs, rollout_act):
    """
    Find the batch probability of observation, action pairs under a given
    policy.

    Params:
    policy (stable_baselines.BaseRLModel): The policy.
    rollout_obs (array) -- A numpy array with shape
      `[n_timesteps] + env.observation_space.shape`.
    rollout_act (array) -- A numpy array with shape
      `[n_timesteps] + env.action_space.shape`.

    Return:
    rollout_prob (array) -- A numpy array with shape `[n_timesteps]`. The
      `i`th entry is the action probability of action `rollout_act[i]` when
      observing `rollout_obs[i]`.
    """

    # TODO: Only tested this on Cartpole (which has discrete actions). No
    # idea how this works in a different action space.
    act_prob = policy.action_probability(rollout_obs)
    if rollout_act.ndim == 1:
        # Expand None dimension so that we can use take_along_axis.
        rollout_act = rollout_act[:, np.newaxis]

    rollout_prob = np.take_along_axis(act_prob, rollout_act, axis=-1)
    rollout_prob = np.squeeze(rollout_prob, axis=1)

    n_timesteps = len(rollout_obs)
    assert len(rollout_obs) == len(rollout_act)
    assert rollout_prob.shape == (n_timesteps,)
    return rollout_prob


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
            tensorboard_log="./output/{}/".format(get_env_id(env)))
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
            policy_class.__name__, get_env_id(env))
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


def generate_rollouts(policy, env, n_timesteps):
    """
    Generate state-action pairs from a policy.

    Params:
    policy (stable_baselines.BaseRLModel) -- A stable_baselines Model, trained
      on the gym environment.
    env (VecEnv or Env or str) -- The environment(s) to interact with.
    n_timesteps (int) -- The number of state-action pairs to collect.

    Return:
    rollout_obs (array) -- A numpy array with shape
      `[n_timesteps] + env.observation_space.shape`.
    rollout_act (array) -- A numpy array with shape
      `[n_timesteps] + env.action_space.shape`.
    """
    if env is None:
        env = policy.env

    env = util.maybe_load_env(env, vectorize=True)
    policy.set_env(env)  # This checks that env and policy are compatbile.
    assert is_vec_env(env)

    rollout_obs = []
    rollout_act = []
    obs = env.reset()
    while len(rollout_obs) < n_timesteps:
        act, _ = policy.predict(obs)
        rollout_obs.extend(obs)
        rollout_act.extend(act)
        obs, _, done, _ = env.step(act)

        # DEBUG
        if np.any(done):
            print("new episode!")

    rollout_obs = np.array(rollout_obs)
    exp_obs = (n_timesteps,) + env.observation_space.shape
    assert rollout_obs.shape == exp_obs

    rollout_act = np.array(rollout_act)
    exp_act = (n_timesteps,) + env.action_space.shape
    assert rollout_act.shape == exp_act

    return rollout_obs, rollout_act
