"""Atari environments wrapped to be properly preprocessed and never terminate.

For final pre-processing, stable_baselines_3.common.vec_env.VecFrameStack should be
applied. (gym's frame_stack produces observations of shape (frames, height, width),
while this codebase expects observaitons of shape (height, width, frames)).

See examples/5a_train_preference_comparisons_with_cnn.ipynb for an example of use.
"""

import gym
from seals.util import AbsorbAfterDoneWrapper
from stable_baselines3.common.atari_wrappers import AtariWrapper


def _make_wrapped_atari(env_id: str) -> gym.Env:
    return AbsorbAfterDoneWrapper(AtariWrapper(gym.make("AsteroidsNoFrameskip-v4")))


def wrapped_asteroids() -> gym.Env:
    return _make_wrapped_atari("AsteroidsNoFrameskip-v4")


def wrapped_beam_rider() -> gym.Env:
    return _make_wrapped_atari("BeamRiderNoFrameskip-v4")


def wrapped_breakout() -> gym.Env:
    return _make_wrapped_atari("BreakoutNoFrameskip-v4")


def wrapped_enduro() -> gym.Env:
    return _make_wrapped_atari("EnduroNoFrameskip-v4")


def wrapped_montezumas_revenge() -> gym.Env:
    return _make_wrapped_atari("MontezumaRevengeNoFrameskip-v4")


def wrapped_pong() -> gym.Env:
    return _make_wrapped_atari("PongNoFrameskip-v4")


def wrapped_qbert() -> gym.Env:
    return _make_wrapped_atari("QbertNoFrameskip-v4")


def wrapped_seaquest() -> gym.Env:
    return _make_wrapped_atari("SeaquestNoFrameskip-v4")


def wrapped_space_invaders() -> gym.Env:
    return _make_wrapped_atari("SpaceInvadersNoFrameskip-v4")
