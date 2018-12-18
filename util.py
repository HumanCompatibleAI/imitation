import os.path
import logging

import gym
import numpy as np
import stable_baselines
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
    env = maybe_load_env(env)
    if is_vec_env(env):
        env = env.envs[0]
    return env.spec.id


def rollout_action_probability(policy, rollout_obs, rollout_act):
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


def make_blank_policy(env, policy_class=stable_baselines.PPO1,
        init_tensorboard=False, policy_network_class="MlpPolicy",
        **kwargs):
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
    return policy_class(policy_network_class, env, verbose=1,
            tensorboard_log=_get_tb_log_dir(env, init_tensorboard),
            **kwargs)


def get_or_train_policy(env, force_train=False, timesteps=500000,
        never_overwrite=False, policy_class=stable_baselines.PPO1,
        init_tensorboard=False, **kwargs):
    """
    Returns a policy trained on the given environment, maybe pretrained.

    If a policy for the environment hasn't been trained and pickled before,
    then first train and pickle it in saved_models/. Otherwise, load that
    pickled policy.

    When looking for trained policies, we first check for an expert policy
    in expert_models/ and then check in saved_models/ for a policy trained
    by this method.

    Params:
    env (str or Env): The Env that this policy is meant to act in, or the
      string name of the Gym environment.
    force_train (bool): If True, then always train a policy, ignoring any
      saved policies.
    timesteps (int): The number of training timesteps if we decide to train
      a policy.
    never_overwrite (bool): It True, then don't pickle a policy if it means
      overwriting another pickle.
    policy_class (stable_baselines.BaseRLModel class): A policy constructor
      from the stable_baselines module.
    init_tensorboard (bool): Whether to initialize Tensorboard logging for
      this policy.
    **kwargs: Additional options for initializing the BaseRLModel class.

    Return:
    policy (stable_baselines.BaseRLModel)
    """
    env = util.maybe_load_env(env)

    if not force_train:
        # Try expert first.
        expert_policy = load_expert_policy(env, policy_class,
                init_tensorboard, **kwargs)
        if expert_policy is not None:
            return expert_policy

        # Try trained.
        trained_policy = load_trained_policy(env, policy_class,
                init_tensorboard, **kwargs)
        if trained_policy is not None:
            return trained_policy

        logging.info("Didn't find a pickled policy. Training...")
    else:
        logging.info("force_train=True. Training...")

    # Learn and pickle a policy
    policy = make_blank_policy(env, policy_class, init_tensorboard, **kwargs)
    policy.learn(timesteps)
    save_trained_policy(policy, never_overwrite)

    return policy


def save_trained_policy(policy, never_overwrite=False):
    """
    Save a trained policy to saved_models/.
    """
    path = os.path.join("saved_models", _policy_filename(policy.__class__))
    if never_overwrite and os.path.exists(path):
        logging.info(("Avoided saving policy pickle to {} because "
            "overwrite is disabled and that file already exists."
                ).format(path))
    else:
        policy.save(path)
        logging.info("Saved pickle to {}!".path)


def load_trained_policy(env, **kwargs):
    """
    Load a trained policy from saved_models/.
    """
    return load_policy(env, basedir="saved_models", **kwargs)


def load_expert_policy(env, **kwargs):
    """
    Load an expert policy from expert_models/.
    """
    return load_policy(env, basedir="expert_models", **kwargs)


# TODO: It's probably a good idea to just use the model/policy semantics
# from stable_baselines, even if I disagree with their naming scheme.
#
# ie, rename policy_class=> policy_model_class
#     rename policy_network_class => policy (matches policy_class.__init__ arg)
def load_policy(env, policy_class=stable_baselines.PPO1, basedir="",
        init_tensorboard=False, policy_network_class=None, **kwargs):
    """
    Load a pickled policy and return it.

    Params:
    env (str or Env): The Env that this policy is meant to act in, or the
      string name of the Gym environment.
    policy_class (stable_baselines.BaseRLModel class): A policy constructor
      from the stable_baselines module.
    init_tensorboard (bool): Whether to initialize Tensorboard logging for
      this policy.
    base_dir (str): The directory of the pickled file.
    policy_network_class (stable_baselines.BasePolicy): A policy network
      constructor. Unless we are using a custom BasePolicy (not builtin to
      stable_baselines), this is automatically infered, and so we can leave
      this argument as None.
    **kwargs: Additional options for initializing the BaseRLModel class.
    """
    path = os.path.join(basedir, _policy_filename(policy_class, env))
    exists = os.path.exists(path)
    if exists:
        env = maybe_load_env(env)
        if policy_network_class is not None:
            kwargs["policy"] = policy_network_class

        policy = policy_class.load(path, env,
                tensorboard_log=_get_tb_log_dir(env, init_tensorboard),
                **kwargs)
        logging.info("loaded policy from '{}'".format(path))
        return policy
    else:
        logging.info("couldn't find policy at '{}'".format(path))
        return None


def _policy_filename(policy_class, env):
    """
    Returns the .pkl filename that the policy instantiated with policy_class
    and trained on env should be saved to.
    """
    return "{}_{}.pkl".format(policy_class.__name__, get_env_id(env))


def generate_rollouts(policy, env, n_timesteps):
    """
    Generate old_obs-action-new_obs-reward tuples from a policy and an
    environment.

    Params:
    policy (stable_baselines.BaseRLModel) -- A stable_baselines Model, trained
      on the gym environment.
    env (VecEnv or Env or str) -- The environment(s) to interact with.
    n_timesteps (int) -- The number of obs-action-obs-reward tuples to collect.

    Return:
    rollout_obs_old (array) -- A numpy array with shape
      `[n_timesteps] + env.observation_space.shape`. The ith observation in this
      array is the observation seen with the agent chooses action
      `rollout_act[i]`.
    rollout_act (array) -- A numpy array with shape
      `[n_timesteps] + env.action_space.shape`.
    rollout_obs_new (array) -- A numpy array with shape
      `[n_timesteps] + env.observation_space.shape`. The ith observation in this
      array is from the transition state after the agent chooses action
      `rollout_act[i]`.
    rollout_rewards (array) -- A numpy array with shape `[n_timesteps]`. The
      reward received on the ith timestep is `rollout_rewards[i]`.
    """
    assert n_timesteps > 0

    env = util.maybe_load_env(env, vectorize=True)
    policy.set_env(env)  # This checks that env and policy are compatbile.
    assert is_vec_env(env)

    rollout_obs_old = []
    rollout_act = []
    rollout_obs_new = []
    rollout_rew = []
    obs = env.reset()
    while len(rollout_obs_new) < n_timesteps:
        # Current state.
        rollout_obs_old.extend(obs)

        # Current action.
        act, _ = policy.predict(obs)
        rollout_act.extend(act)

        # Transition state.
        obs, _, done, rew = env.step(act)
        rollout_obs_new.extend(obs)

        # Rewards.
        rollout_rew.extend(rew)

        if np.any(done):
            logging.debug("new episode!")

    rollout_obs_new = np.array(rollout_obs_new)[:n_timesteps]
    rollout_obs_old = np.array(rollout_obs_old)[:n_timesteps]
    rollout_rew = np.array(rollout_rew)[:n_timesteps]
    exp_obs = (n_timesteps,) + env.observation_space.shape
    assert rollout_obs_new.shape == exp_obs
    assert rollout_obs_old.shape == exp_obs
    assert rollout_rew.shape == (n_timesteps,)

    rollout_act = np.array(rollout_act)[:n_timesteps]
    exp_act = (n_timesteps,) + env.action_space.shape
    assert rollout_act.shape == exp_act
    assert np.all(rollout_obs_new[:-1] == rollout_obs_old[1:])

    return rollout_obs_old, rollout_act, rollout_obs_new, rollout_rew


def _get_tb_log_dir(env, init_tensorboard):
    if init_tensorboard:
        return "./output/{}/".format(get_env_id(env))
    else:
        return None
