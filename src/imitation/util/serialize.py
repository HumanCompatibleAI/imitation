"""Helper classes to serialize classes with TensorFlow objects."""

from abc import ABC, abstractmethod
import os
import pickle
from typing import Type, TypeVar

import tensorflow as tf

from imitation.util import util


def make_cls(cls, args, kwargs):
  return cls(*args, **kwargs)


class Serializable(ABC):
  """Abstract mix-in defining methods to load/save model."""
  @classmethod
  @abstractmethod
  def load(cls, directory):
    """Load object plus weights from directory."""

  @abstractmethod
  def save(self, directory):
    """Save object and weights to directory."""


T = TypeVar('T')


class LayersSerializable(Serializable):
  """Serialization mix-in based on `__init__` then rehydration.

  Subclsases must call the constructor with all arguments needed by `__init__`,
  and a dictionary mapping from strings to `tf.layers.Layer` objects.
  In most cases, you can use the following idiom::

      args = locals()
      # ... your normal constructor
      layers = # ... gather your TensorFlow objects
      LayersSerializable.__init__(**args, layers=layers)
  """

  def __init__(self, *args, layers: util.LayersDict, **kwargs):
    self._args = args
    self._kwargs = kwargs
    self._checkpoint = tf.train.Checkpoint(**layers)

  def __reduce__(self):
    return (make_cls, (type(self), self._args, self._kwargs))

  def save_parameters(self, directory: str) -> None:
    self._checkpoint.save(file_prefix=os.path.join(directory, "weights"))

  def load_parameters(self, directory: str) -> None:
    restore = self._checkpoint.restore(tf.train.latest_checkpoint(directory))
    restore.assert_consumed().run_restore_ops()

  @classmethod
  def load(cls: Type[T], directory: str) -> T:
    with open(os.path.join(directory, 'obj'), 'rb') as f:
      obj = pickle.load(f)
    assert isinstance(obj, cls)
    obj.load_parameters(directory)
    return obj

  def save(self, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

    obj_path = os.path.join(directory, 'obj')
    if not os.path.exists(obj_path):
      # TODO(gleave): it'd be nice to support Python state changing too
      # obj should be static (since it just consists of _args and _kwargs,
      # set at construction). So no need to write this file multiple times.
      # (In fact, best to avoid it -- if we were to die in the middle of this,
      # it would invalidate previous checkpoints.)
      with open(obj_path, 'wb') as f:
        pickle.dump(self, f)

    self._checkpoint.save(file_prefix=os.path.join(directory, "weights"))
