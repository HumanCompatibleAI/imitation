import collections
import contextlib
import datetime
import functools
import os
from typing import Callable, Dict, Iterable, Optional, Tuple, Type, Union

import gym
import stable_baselines
from stable_baselines import bench
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.input import observation_input
from stable_baselines.common.policies import BasePolicy, MlpPolicy
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


def init_rl(env: Union[gym.Env, VecEnv],
            model_class: Type[BaseRLModel] = stable_baselines.PPO2,
            policy_class: Type[BasePolicy] = MlpPolicy,
            **model_kwargs):
  """Instantiates a policy for the provided environment.

  Args:
      env: The (vector) environment.
      model_class: A Stable Baselines RL algorithm.
      policy_class: A Stable Baselines compatible policy network class.
      model_kwargs (dict): kwargs passed through to the algorithm.
        Note: anything specified in `policy_kwargs` is passed through by the
        algorithm to the policy network.

  Returns:
    An RL algorithm.
  """
  return model_class(policy_class, env, **model_kwargs)


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
                            kernel_initializer=initializer,
                            name=key)  # type: tf.layers.Layer
    layers[key] = layer

  # Final layer
  layer = tf.layers.Dense(1, kernel_initializer=initializer,
                          name=f"{name}_dense_final")  # type: tf.layers.Layer
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


@contextlib.contextmanager
def make_session(**kwargs):
  """Context manager for a TensorFlow session.

  The session is associated with a newly created graph. Both session and
  graph are set as default. The session will be closed when exiting this
  context manager.

  Args:
    kwargs: passed through to `tf.Session`.

  Yields:
    (graph, session) where graph is a `tf.Graph` and `session` a `tf.Session`.
  """
  graph = tf.Graph()
  with graph.as_default():
    with tf.Session(graph=graph, **kwargs) as session:
      with session.as_default():
        yield graph, session
