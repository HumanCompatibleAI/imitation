# Add our custom environments to Gym registry.

try:
    # pytype: disable=import-error
    import seals  # noqa: F401

    # pytype: enable=import-error
except ImportError:
    pass

import imitation.envs.examples  # noqa: F401
