import logging
import os

import gym
import stable_baselines
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecEnv


def maybe_load_env(env_or_str, vectorize=True):
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


def make_vec_env(env_id, n_envs=8):
    """
    Make a DummyVecEnv initialized with `n_envs` Envs.

    Params:
    env_id (str): The Env's string id in Gym.
    n_envs (int): The number of duplicate environments.
    """
    return DummyVecEnv([lambda: gym.make(env_id) for _ in range(n_envs)])


def is_vec_env(env):
    return isinstance(env, VecEnv)


def get_env_id(env):
    try:
        env = maybe_load_env(env)
        if is_vec_env(env):
            env = env.envs[0]
        return env.spec.id
    except Exception as e:
        logging.warning("Couldn't find environment id, using 'UnknownEnv'")
        logging.warning(e)
        return "UnknownEnv"


class _FeedForward32Policy(FeedForwardPolicy):
    """
    A feed forward gaussian policy network with two hidden layers of 32 units.
    This matches the IRL policies in the original AIRL paper.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                net_arch=[32, 32], feature_extraction="mlp")


def make_blank_policy(env, policy_class=stable_baselines.PPO2,
        init_tensorboard=False, policy_network_class=_FeedForward32Policy,
        verbose=0, **kwargs):
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
    verbose (int): The verbosity level of the policy during training.

    Return:
    policy (stable_baselines.BaseRLModel)
    """
    env = maybe_load_env(env)
    return policy_class(policy_network_class, env, verbose=verbose,
            tensorboard_log=_get_tb_log_dir(env, init_tensorboard),
            **kwargs)


def get_or_train_policy(env, force_train=False, timesteps=500000,
        policy_class=stable_baselines.PPO2,
        init_tensorboard=False, verbose=0, **kwargs):
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
    policy_class (stable_baselines.BaseRLModel class): A policy constructor
      from the stable_baselines module.
    init_tensorboard (bool): Whether to initialize Tensorboard logging for
      this policy.
    verbose (int): The verbosity level of the policy during training.
    **kwargs: Additional options for initializing the BaseRLModel class.

    Return:
    policy (stable_baselines.BaseRLModel)
    """
    env = maybe_load_env(env)

    if not force_train:
        # Try expert first.
        expert_policy = load_expert_policy(env, policy_class,
                init_tensorboard, verbose=verbose, **kwargs)
        if expert_policy is not None:
            return expert_policy

        # Try trained.
        trained_policy = load_trained_policy(env, policy_class,
                init_tensorboard, verbose=verbose, **kwargs)
        if trained_policy is not None:
            return trained_policy

        logging.info("Didn't find a pickled policy. Training...")
    else:
        logging.info("force_train=True. Training...")

    # Learn and pickle a policy
    policy = make_blank_policy(env, policy_class, init_tensorboard, **kwargs)
    policy.learn(timesteps)
    save_trained_policy(policy)

    return policy


def save_trained_policy(policy, savedir="saved_models", filename=None):
    """
    Save a trained policy as a pickle file.

    Params:
    savedir (str): The directory to save the file to.
    filename (str): The the name of the pickle file. If None, then choose
      a default name using the names of the policy model and the environment.
    """
    filename = filename or _policy_filename(policy.__class__, env)
    os.makedirs(savedir)
    path = os.path.join(savedir, filename)
    policy.save(path)
    logging.info("Saved pickle to {}!".path)


def make_save_policy_callback(savedir, file_prefix, save_interval):
    """
    Make a policy.learn() callback that saves snapshots of the policy
    to `{savedir}/{save_prefix}-{step}`, where step is the training
    step.

    Params:
    savedir (str): The directory to save in.
    save_prefix (str): The pickle file prefix.
    save_interval (int): The number of training timesteps in between saves.
    """
    step = 0
    save_prefix = save_prefix
    def callback(locals_, globals_):
        step += 1
        if step % save_interval == 0:
            policy = locals_['self']
            # TODO: After we use globs in scripts.data_generate_...,
            # then we can simply use step.
            filename = "{}-{}".format(file_prefix, step//step_interval)
            save_trained_policy(policy, savedir, filename)
        return True
    return callback


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
def load_policy(env, policy_class=stable_baselines.PPO2, basedir="",
        init_tensorboard=False, policy_network_class=None, verbose=0, **kwargs):
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
    verbose (int): The verbosity level of the model during training.
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


def _get_tb_log_dir(env, init_tensorboard):
    if init_tensorboard:
        return "./output/{}/".format(get_env_id(env))
    else:
        return None
