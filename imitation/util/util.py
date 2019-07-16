import collections
import datetime
import functools
import glob
import os
from typing import Callable, Dict, Iterable, Optional, Tuple

import gym
import stable_baselines
from stable_baselines import bench
from stable_baselines.common.input import observation_input
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
import tensorflow as tf

# TODO(adam): this should really be OrderedDict but that breaks Python
# See https://stackoverflow.com/questions/41207128/
LayersDict = Dict[str, tf.layers.Layer]


def make_timestamp():
  ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
  return datetime.datetime.now().strftime(ISO_TIMESTAMP)


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


def make_vec_env(env_id: str,
                 n_envs: int = 8,
                 seed: int = 0,
                 parallel: bool = False,
                 log_dir: Optional[str] = None) -> VecEnv:
  """Returns a VecEnv initialized with `n_envs` Envs.

  Args:
      env_id: The Env's string id in Gym.
      n_envs: The number of duplicate environments.
      seed: The environment seed.
      parallel: If True, uses SubprocVecEnv; otherwise, DummyVecEnv.
      log_dir: If specified, saves Monitor output to this directory.
  """
  def make_env(i):
    env = gym.make(env_id)
    env.seed(seed + i)  # seed each environment separately for diversity

    # Use Monitor to record statistics needed for Baselines algorithms logging
    # Optionally, save to disk
    log_path = None
    if log_dir is not None:
      log_subdir = os.path.join(log_dir, 'monitor')
      os.makedirs(log_subdir, exist_ok=True)
      log_path = os.path.join(log_subdir, f'mon{i:03d}')
    return bench.Monitor(env, log_path, allow_early_resets=True)
  env_fns = [functools.partial(make_env, i) for i in range(n_envs)]
  if parallel:
    # See GH hill-a/stable-baselines issue #217
    return SubprocVecEnv(env_fns, start_method='forkserver')
  else:
    return DummyVecEnv(env_fns)


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
  """A feed forward policy network with two hidden layers of 32 units.

  This matches the IRL policies in the original AIRL paper.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs,
                     net_arch=[32, 32], feature_extraction="mlp")


class FeedForward64Policy(FeedForwardPolicy):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs,
                     net_arch=[64, 64], feature_extraction="mlp")


def make_blank_policy(env, policy_class=stable_baselines.PPO2,
                      policy_network_class=FeedForward32Policy, verbose=1,
                      **kwargs):
  """Instantiates a policy for the provided environment.

  Args:
      env (str or Env): The Env or its string id in Gym.
      policy_network_class (stable_baselines.BasePolicy): A policy network
          constructor from the stable_baselines module.
      policy_class (stable_baselines.BaseRLModel subclass): A policy constructor
          from the stable_baselines module.
      verbose (int): The verbosity level of the policy during training.

  Return:
  policy (stable_baselines.BaseRLModel)
  """
  env = maybe_load_env(env)
  return policy_class(policy_network_class, env, verbose=verbose, **kwargs)


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
      save_trained_policy(policy, savedir, f'{step:05d}')
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


def load_policy(env, basedir="expert_models",
                policy_model_class=stable_baselines.PPO2,
                policy_network_class=None, n_experts=1,
                **kwargs):
  """Loads and returns a pickled policy.

  Args:
      env (str): The string name of the Gym environment the policy is acting in.
      policy_class (stable_baselines.BaseRLModel class): A policy constructor
          from the stable_baselines module.
      base_dir (str): The directory of the pickled file.
      policy_network_class (stable_baselines.BasePolicy): A policy network
          constructor. Unless we are using a custom BasePolicy (not builtin to
          stable_baselines), this is automatically infered, and so we can leave
          this argument as None.
      **kwargs: Additional options for initializing the BaseRLModel class.
  """

  # FIXME: Despite name, this does not actually load policies, it loads lists
  # of pickled policy training algorithms ("RL models" in stable-baselines'
  # terminology). Should fix the naming, or change it so that it actually loads
  # policies (which is often what is really wanted, IMO).

  # FIXME: a lot of code assumes that this function returns None on failure,
  # which it does not. That upstream code also needs to be fixed.

  paths = _get_policy_paths(env, policy_model_class, basedir, n_experts)

  env = maybe_load_env(env)

  if (policy_network_class is not None) and ("policy" not in kwargs):
    kwargs["policy"] = policy_network_class

  pols = []

  for path in paths:
    policy = policy_model_class.load(path, env, **kwargs)
    tf.logging.info("loaded policy from '{}'".format(path))
    pols.append(policy)

  return pols


def _policy_filename(policy_class, env, n="[0-9]*"):
  """Returns the .pkl filename that the policy instantiated with `policy_class`
  and trained on env should be saved to.
  """
  return "{}_{}_{}.pkl".format(policy_class.__name__, get_env_id(env), n)


def build_mlp(hid_sizes: Iterable[int],
              name: Optional[str] = None,
              activation: Optional[Callable] = tf.nn.relu,
              initializer: Optional[Callable] = None,
              ) -> LayersDict:
  """Constructs an MLP, returning an ordered dict of layers."""
  layers = collections.OrderedDict()

  # Hidden layers
  for i, size in enumerate(hid_sizes):
    key = f"{name}_dense{i}"
    layer = tf.layers.Dense(size, activation=activation,
                            kernel_initializer=initializer, name=key)
    layers[key] = layer

  # Final layer
  layer = tf.layers.Dense(1, kernel_initializer=initializer,
                          name=f"{name}_dense_final")
  layers[f"{name}_dense_final"] = layer

  return layers


def sequential(inputs: tf.Tensor,
               layers: LayersDict,
               ) -> tf.Tensor:
  """Applies a sequence of layers to an input."""
  output = inputs
  for layer in layers.values():
    output = layer(output)
  output = tf.squeeze(output, axis=1)
  return output


def build_inputs(observation_space: gym.Space,
                 action_space: gym.Space,
                 scale: bool = False) -> Tuple[tf.Tensor, ...]:
  """Builds placeholders and processed input Tensors.

  Observation `old_obs_*` and `new_obs_*` placeholders and processed input
  tensors have shape `(None,) + obs_space.shape`.
  The action `act_*` placeholder and processed input tensors have shape
  `(None,) + act_space.shape`.

  Args:
    observation_space: The observation space.
    action_space: The action space.
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
  old_obs_ph, old_obs_inp = observation_input(observation_space,
                                              name="old_obs", scale=scale)
  act_ph, act_inp = observation_input(action_space, name="act", scale=scale)
  new_obs_ph, new_obs_inp = observation_input(observation_space,
                                              name="new_obs", scale=scale)
  return old_obs_ph, act_ph, new_obs_ph, old_obs_inp, act_inp, new_obs_inp
