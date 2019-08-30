"""Utilities for distributed experiments."""
import ray.tune.track


def ray_tune_active() -> bool:
  """Returns True if a Ray Tune track session has been initialized."""
  try:
    return ray.tune.track.get_session() is not None
  except ValueError:
    return False
