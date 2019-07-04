import glob
import os
from typing import Iterable, Optional, Tuple

import gin
import gin.tf
import gym
import stable_baselines
from stable_baselines.bench import Monitor
from stable_baselines.common.input import observation_input
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecEnv
import tensorflow as tf


def maybe_load_env(env_or_str, vectorize=True):
  """Load an environment if it isn't already loaded. Then optionally vectorize
  it as a DummyVecEnv, if it isn't already vectorized.

  Args:
      env_or_str (str or gym.Env): If `env_or_str` is a str, it's loaded
          before returning.
      vectorize (bool): If True, then vectorize the environment before
          returning, if it isn't already vectorized.

  Return:
      env (gym.Env): Either the original argument if it was an Env or an
          instantiated gym Env if it was a string.
  """
  if isinstance(env_or_str, str):
    env = gym.make(env_or_str)
  else:
    env = env_or_str

  if not is_vec_env(env) and vectorize:
    env = DummyVecEnv([lambda: env])

  return env


def make_vec_env(env_id, n_envs=8):
  """Returns a DummyVecEnv initialized with `n_envs` Envs.

  Args:
      env_id (str): The Env's string id in Gym.
      n_envs (int): The number of duplicate environments.
  """
  # Use Monitor to support logging the episode reward and length.
  def monitored_env():
    return Monitor(gym.make(env_id), None, allow_early_resets=True)
  return DummyVecEnv([monitored_env for _ in range(n_envs)])


def is_vec_env(env):
  return isinstance(env, VecEnv)


def get_env_id(env_or_str):
  if isinstance(env_or_str, str):
    return env_or_str

  try:
    env = maybe_load_env(env_or_str)
    if is_vec_env(env):
      env = env.envs[0]
    return env.spec.id
  except Exception as e:
    tf.logging.warning("Couldn't find environment id, using 'UnknownEnv'")
    tf.logging.warning(e)
    return "UnknownEnv"


class FeedForward32Policy(FeedForwardPolicy):
  """A feed forward gaussian policy network with two hidden layers of 32 units.

  This matches the IRL policies in the original AIRL paper.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs,
                     net_arch=[32, 32], feature_extraction="mlp")


@gin.configurable
def make_blank_policy(env, policy_class=stable_baselines.PPO2,
                      init_tensorboard=False,
                      policy_network_class=FeedForward32Policy, verbose=0,
                      **kwargs):
  """Instantiates a policy for the provided environment.

  Args:
      env (str or Env): The Env or its string id in Gym.
      policy_network_class (stable_baselines.BasePolicy): A policy network
          constructor from the stable_baselines module.
      policy_class (stable_baselines.BaseRLModel subclass): A policy constructor
          from the stable_baselines module.
      init_tensorboard (bool): Whether to make TensorBoard summary writes during
          training.
      verbose (int): The verbosity level of the policy during training.

  Return:
  policy (stable_baselines.BaseRLModel)
  """
  env = maybe_load_env(env)
  tf.logging.info("kwargs %s", kwargs)
  return policy_class(policy_network_class, env, verbose=verbose,
                      tensorboard_log=_get_tb_log_dir(env, init_tensorboard),
                      **kwargs)


def save_trained_policy(policy, savedir="saved_models", filename=None):
  """Saves a trained policy as a pickle file.

  Args:
      policy (BasePolicy): policy to save
      savedir (str): The directory to save the file to.
      filename (str): The the name of the pickle file. If None, then choose
          a default name using the names of the policy model and the
          environment.
  """
  os.makedirs(savedir, exist_ok=True)
  path = os.path.join(savedir, filename)
  policy.save(path)
  tf.logging.info("Saved pickle to {}!".format(path))


def make_save_policy_callback(savedir, save_interval=1):
  """Makes a policy.learn() callback that saves snapshots of the policy
  to `{savedir}/{file_prefix}-{step}`, where step is the training
  step.

  Args:
      savedir (str): The directory to save in.
      file_prefix (str): The pickle file prefix.
      save_interval (int): The number of training timesteps in between saves.
  """
  step = 0

  def callback(locals_, _):
    nonlocal step
    step += 1
    if step % save_interval == 0:
      policy = locals_['self']
      filename = _policy_filename(policy.__class__, policy.get_env(),
                                  step)
      save_trained_policy(policy, savedir, filename)
    return True

  return callback


def _get_policy_paths(env, policy_model_class, basedir, n_experts):
  assert n_experts > 0

  path = os.path.join(basedir, _policy_filename(policy_model_class, env))
  paths = glob.glob(path)

  paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

  if len(paths) < n_experts:
    raise ValueError(
        "Wanted to load {} experts, but there were only {} experts at {}"
        .format(n_experts, len(paths), path))

  paths = paths[-n_experts:]

  return paths


@gin.configurable
def load_policy(env, basedir, policy_model_class=stable_baselines.PPO2,
                init_tensorboard=False, policy_network_class=None, n_experts=1,
                **kwargs):
  """Loads and returns a pickled policy.

  Args:
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

  paths = _get_policy_paths(env, policy_model_class, basedir, n_experts)

  env = maybe_load_env(env)

  if (policy_network_class is not None) and ("policy" not in kwargs):
    kwargs["policy"] = policy_network_class

  pols = []

  for path in paths:
    policy = policy_model_class.load(
        path, env, tensorboard_log=_get_tb_log_dir(env, init_tensorboard),
        **kwargs)
    tf.logging.info("loaded policy from '{}'".format(path))
    pols.append(policy)

  return pols


def _policy_filename(policy_class, env, n="[0-9]*"):
  """Returns the .pkl filename that the policy instantiated with `policy_class`
  and trained on env should be saved to.
  """
  return "{}_{}_{}.pkl".format(policy_class.__name__, get_env_id(env), n)


def _get_tb_log_dir(env, init_tensorboard):
  if init_tensorboard:
    return "./output/{}/".format(get_env_id(env))
  else:
    return None


def apply_ff(inputs: tf.Tensor,
             hid_sizes: Iterable[int],
             name: Optional[str] = None,
             ) -> tf.Tensor:
  """Applies a feed forward network on the inputs."""
  xavier = tf.contrib.layers.xavier_initializer
  x = inputs
  for i, size in enumerate(hid_sizes):
    x = tf.layers.dense(x, size, activation='relu',
                        kernel_initializer=xavier(), name="dense" + str(i))
  x = tf.layers.dense(x, 1, kernel_initializer=xavier(),
                      name="dense_final")
  return tf.squeeze(x, axis=1, name=name)


def build_inputs(env: gym.Env, scale: bool = True) -> Tuple[tf.Tensor, ...]:
  """Build placeholders and processed input Tensors for `old_obs`, `act`, and
  `new_obs`.

  Observation placeholders and processed input tensors have shape
  `(None,) + obs_space.shape`.
  The action placeholder and processed input tensors have shape
  `(None,) + act_space.shape`.

  Args:
    env: The environment that the observation and action spaces comes from.
    scale: Only relevant for environments with Box spaces. If True, then
      processed input Tensors are automatically scaled to the interval [0, 1].

  Returns:
    old_obs_ph: Placeholder for old observations.
    act_ph: Placeholder for actions.
    new_obs_ph: Placeholder for new observations.
    old_obs_inp: Network-ready float32 Tensor with processed old observations.
    act_inp: Network-ready float32 Tensor with processed actions.
    new_obs_inp: Network-ready float32 Tensor with processed new observations.
  """

  old_obs_ph, old_obs_inp = observation_input(env.observation_space,
                                              name="old_obs", scale=scale)
  act_ph, act_inp = observation_input(env.action_space, name="act", scale=scale)
  new_obs_ph, new_obs_inp = observation_input(env.observation_space,
                                              name="new_obs", scale=scale)
  return old_obs_ph, act_ph, new_obs_ph, old_obs_inp, act_inp, new_obs_inp
