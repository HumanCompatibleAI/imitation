"""Utilities for distributed experiments."""
import functools
from pprint import pprint as pp
from typing import Callable, Optional

import ray
import ray.tune as tune
import sacred


def ray_tune_active() -> bool:
  """Returns True if a Ray Tune track session has been initialized."""
  try:
    return ray.tune.track.get_session() is not None
  except ValueError:
    return False
