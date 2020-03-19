# Add our custom environments to Gym registry.

try:
  import benchmark_environments.classic_control  # noqa: F401
except ImportError:
  pass

import imitation.envs.examples  # noqa: F401
