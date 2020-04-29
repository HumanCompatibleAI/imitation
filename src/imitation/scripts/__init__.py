# Add our custom environments to Gym registry.

try:
    # pytype: disable=import-error
    import benchmark_environments.classic_control  # noqa: F401

    # pytype: enable=import-error
except ImportError:
    pass

import imitation.envs.examples  # noqa: F401
