import importlib
from typing import Callable, Generic, Optional, Type, TypeVar

from stable_baselines.common.vec_env import VecEnv

T = TypeVar("T")
LoaderFn = Callable[[str, VecEnv], T]
"""The type stored in Registry is commonly an instance of LoaderFn."""


def load(name):
  """Load an attribute in format path.to.module:attribute."""
  module_name, attr_name = name.split(":")
  module = importlib.import_module(module_name)
  attr = getattr(module, attr_name)
  return attr


class Registry(Generic[T]):
  """A registry mapping IDs to type T objects, with support for lazy loading.

  The registry allows for insertion and retrieval. Modification of existing
  elements is not allowed.

  If the registered item is a string, it is assumed to be a path to an attribute
  in the form path.to.module:attribute. In this case, the module is loaded
  only if and when the registered item is retrieved.

  This is helpful both to reduce overhead from importing unused modules,
  and when some modules may have additional dependencies that are not installed
  in all deployments.

  Note: This is a similar idea to gym.EnvRegistry.
  """

  def __init__(self):
    self._values = {}
    self._indirect = {}

  def get(self, key: str) -> T:
    if key not in self._values and key not in self._indirect:
      raise KeyError(f"Key '{key}' is not registered.")

    if key not in self._values:
      self._values[key] = load(self._indirect[key])
    return self._values[key]

  def register(self, key: str,
               value: Optional[T] = None,
               indirect: Optional[str] = None):
    if key in self._values or key in self._indirect:
      raise KeyError(f"Duplicate registration for '{key}'")

    provided_args = sum([value is not None, indirect is not None])
    if provided_args != 1:
      raise ValueError("Must provide exactly one of 'value' and 'indirect',"
                       f"{provided_args} have been provided.")

    if value is not None:
      self._values[key] = value
    else:
      self._indirect[key] = indirect


def env_to_space(cls: Type[T], **kwargs) -> LoaderFn:
  """Converts a factory taking observation and action space into a LoaderFn."""
  def f(path: str, venv: VecEnv) -> T:
    return cls(venv.observation_space, venv.action_space, **kwargs)
  return f


def env_only(policy_cls: Type[T], **kwargs) -> LoaderFn:
  """Converts a factory taking an environment into a LoaderFn."""
  def f(path: str, venv: VecEnv) -> T:
    return policy_cls(venv, **kwargs)
  return f
