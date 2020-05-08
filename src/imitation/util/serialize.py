"""Helper classes to serialize classes with TensorFlow objects."""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Type, TypeVar

import tensorflow as tf

from imitation.util import networks

T = TypeVar("T", bound="Serializable")


class Serializable(ABC):
    """Abstract mix-in defining methods to load/save model."""

    @classmethod
    def load(cls: Type[T], directory: str) -> T:
        """Load object plus weights from directory."""
        with open(os.path.join(directory, "loader"), "rb") as f:
            load_cls = pickle.load(f)
        return load_cls._load(directory)

    def save(self, directory: str) -> None:
        """Save object and weights to directory."""
        os.makedirs(directory, exist_ok=True)

        load_path = os.path.join(directory, "loader")
        with open(load_path + ".tmp", "wb") as f:
            pickle.dump(type(self), f)
        # Ensure atomic write
        os.replace(load_path + ".tmp", load_path)

        self._save(directory)

    @classmethod
    @abstractmethod
    def _load(cls: Type[T], directory: str) -> T:
        """Class-specific loading logic."""

    @abstractmethod
    def _save(self, directory: str) -> None:
        """Class-specific saving logic."""


def make_cls(cls, args, kwargs):
    return cls(*args, **kwargs)


class LayersSerializable(Serializable):
    """Serialization mix-in based on `__init__` then rehydration.

    Subclasses must call the constructor with all arguments needed by `__init__`,
    and a dictionary mapping from strings to `tf.layers.Layer` objects.
    In most cases, you can use the following idiom::

        args = locals()
        # ... your normal constructor
        layers = # ... gather your TensorFlow objects
        LayersSerializable.__init__(**args, layers=layers)
    """

    def __init__(self, *args, layers: networks.LayersDict, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._checkpoint = tf.train.Checkpoint(**layers)

    def __reduce__(self):
        return (make_cls, (type(self), self._args, self._kwargs))

    def save_parameters(self, directory: str) -> None:
        file_prefix = os.path.join(directory, "weights")
        # TensorFlow stores a path to the latest checkpoint. It will use a relative
        # path if you give a relative `file_prefix`, otherwise it will use
        # absolute paths. We want it to use relative paths so saved models are
        # portable across machines. See TF issue #2973.
        file_prefix = os.path.relpath(file_prefix)
        self._checkpoint.save(file_prefix=file_prefix)

    def load_parameters(self, directory: str) -> None:
        restore = self._checkpoint.restore(tf.train.latest_checkpoint(directory))
        restore.assert_consumed().run_restore_ops()

    @classmethod
    def _load(cls: Type[T], directory: str) -> T:
        with open(os.path.join(directory, "obj"), "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, cls)
        obj.load_parameters(directory)
        return obj

    def _save(self, directory: str) -> None:
        obj_path = os.path.join(directory, "obj")
        # obj is static, so no need to write them multiple times.
        # (Best to avoid it -- if we were die in the middle of save, it could
        # cause data corruption invalidating previous checkpoints.)
        if not os.path.exists(obj_path):
            # TODO(gleave): it'd be nice to support mutable Python state
            with open(obj_path, "wb") as f:
                pickle.dump(self, f)

        self.save_parameters(directory)
