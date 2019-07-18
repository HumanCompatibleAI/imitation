import functools
import os
from typing import Iterable, List, Optional, Tuple, Union

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


# TODO(adam): performance enhancement -- make Dummy vs Subproc configurable
def make_vec_env(env_id: str, n_envs: int = 8, seed: int = 0):
  """Returns a DummyVecEnv initialized with `n_envs` Envs.

  Args:
      env_id: The Env's string id in Gym.
      n_envs: The number of duplicate environments.
      seed: The environment seed.
  """
  def monitored_env(i):
    env = gym.make(env_id)
    env.seed(seed + i)
    # Use Monitor to support logging the episode reward and length.
    # TODO(adam): also log to disk -- can be useful for debugging occasionally?
    return Monitor(env, None, allow_early_resets=True)
  return DummyVecEnv([functools.partial(monitored_env, i)
                      for i in range(n_envs)])


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
                      init_tensorboard=False,
                      policy_network_class=FeedForward32Policy, verbose=1,
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
  return policy_class(policy_network_class, env, verbose=verbose,
                      tensorboard_log=_get_tb_log_dir(env, init_tensorboard),
                      **kwargs)


def load_policy(path: str,
                env: gym.Env,
                policy_model_class=stable_baselines.PPO2,
                init_tensorboard=False,
                policy_network_class=None,
                **kwargs) -> stable_baselines.BaseRLModel:
  """Loads and returns an policy encapsulated in a RLModel.

  Args:
      path (str): Path to the policy dump.
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

  # FIXME: Despite name, this does not actually load policies, it loads lists
  # of pickled policy training algorithms ("RL models" in stable-baselines'
  # terminology). Should fix the naming, or change it so that it actually loads
  # policies (which is often what is really wanted, IMO).

  env = maybe_load_env(env)

  if (policy_network_class is not None) and ("policy" not in kwargs):
    kwargs["policy"] = policy_network_class

  policy = policy_model_class.load(path, env, **kwargs)
  tf.logging.info("loaded policy from '{}'".format(path))
  return policy


def dump_prefix(policy_class, env, n: Union[int, str]) -> str:
  """Build the standard filename prefix of .pkl and .npz dumps.

  Args:
      policy_class (stable_baselines.BaseRLModel subclass): The policy class.
      env: The environment.
      n: Either the training step number, or a glob expression for matching dump
          files in `get_dump_paths`.
  """
  return "{}_{}_{}".format(get_env_id(env), policy_class.__name__, n)


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
