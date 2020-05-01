"""Environments used for testing and benchmarking.

These are not a core part of the imitation package. They are relatively lightly tested,
and may be changed without warning.
"""

# Register environments with Gym
from imitation.envs.examples import airl_envs, model_envs  # noqa: F401
