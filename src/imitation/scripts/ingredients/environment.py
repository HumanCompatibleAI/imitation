"""This ingredient provides a vectorized gym environment."""
import contextlib
import functools
from typing import Any, Callable, Generator, Mapping

import gym
import numpy as np
import sacred
from stable_baselines3.common import vec_env

from imitation.data import wrappers
from imitation.util import util

environment_ingredient = sacred.Ingredient("environment")


@environment_ingredient.config
def config():
    num_vec = 8  # number of environments in VecEnv
    parallel = True  # Use SubprocVecEnv rather than DummyVecEnv
    max_episode_steps = None  # Set to positive int to limit episode horizons
    env_make_kwargs = {}  # The kwargs passed to `spec.make`.
    gym_id = "seals/CartPole-v0"  # The environment to train on
    post_wrappers = {}  # Wrappers applied after `spec.make`
    post_wrappers_kwargs = {}  # The kwargs passed to post wrappers

    locals()  # quieten flake8


@contextlib.contextmanager
@environment_ingredient.capture
def make_venv(
    gym_id: str,
    num_vec: int,
    parallel: bool,
    max_episode_steps: int,
    env_make_kwargs: Mapping[str, Any],
    post_wrappers: Mapping[str, Callable[[gym.Env, int], gym.Env]],
    post_wrappers_kwargs: Mapping[str, Mapping[str, Any]],
    _run: sacred.run.Run,
    _rnd: np.random.Generator,
    **kwargs,
) -> Generator[vec_env.VecEnv, None, None]:
    """Builds the vector environment.

    Args:
        gym_id: The id of the environment to create.
        num_vec: Number of `gym.Env` instances to combine into a vector environment.
        parallel: Whether to use "true" parallelism. If True, then use `SubProcVecEnv`.
            Otherwise, use `DummyVecEnv` which steps through environments serially.
        max_episode_steps: If not None, then a TimeLimit wrapper is applied to each
            environment to artificially limit the maximum number of timesteps in an
            episode.
        env_make_kwargs: The kwargs passed to `spec.make` of a gym environment.
        post_wrappers: The wrappers applied after environment creation with `spec.make`.
        post_wrappers_kwargs: List of kwargs passed to the respective post wrappers.
        kwargs: Passed through to `util.make_vec_env`.

    Yields:
        The constructed vector environment.
    """
    # Update env_fns for post wrappers with kwargs
    updated_post_wrappers = []
    for key, post_wrapper in post_wrappers.items():
        if key in post_wrappers_kwargs:
            post_wrapper = functools.partial(post_wrapper, **post_wrappers_kwargs[key])
        updated_post_wrappers.append(post_wrapper)

    # Note: we create the venv outside the try -- finally block for the case that env
    #     creation fails.
    venv = util.make_vec_env(
        gym_id,
        rng=_rnd,
        n_envs=num_vec,
        parallel=parallel,
        max_episode_steps=max_episode_steps,
        log_dir=_run.config["logging"]["log_dir"] if "logging" in _run.config else None,
        env_make_kwargs=env_make_kwargs,
        post_wrappers=updated_post_wrappers,
        **kwargs,
    )
    try:
        yield venv
    finally:
        venv.close()


@contextlib.contextmanager
@environment_ingredient.capture
def make_rollout_venv(
    gym_id: str,
    num_vec: int,
    parallel: bool,
    max_episode_steps: int,
    env_make_kwargs: Mapping[str, Any],
    _rnd: np.random.Generator,
) -> Generator[vec_env.VecEnv, None, None]:
    """Builds the vector environment for rollouts.

    This environment does no logging, and it is wrapped in a `RolloutInfoWrapper`.

    Args:
        gym_id: The id of the environment to create.
        num_vec: Number of `gym.Env` instances to combine into a vector environment.
        parallel: Whether to use "true" parallelism. If True, then use `SubProcVecEnv`.
            Otherwise, use `DummyVecEnv` which steps through environments serially.
        max_episode_steps: If not None, then a TimeLimit wrapper is applied to each
            environment to artificially limit the maximum number of timesteps in an
            episode.
        env_make_kwargs: The kwargs passed to `spec.make` of a gym environment.
        _rnd: Random number generator provided by Sacred.

    Yields:
        The constructed vector environment.
    """
    # Note: we create the venv outside the try -- finally block for the case that env
    #     creation fails.
    venv = util.make_vec_env(
        gym_id,
        rng=_rnd,
        n_envs=num_vec,
        parallel=parallel,
        max_episode_steps=max_episode_steps,
        log_dir=None,
        env_make_kwargs=env_make_kwargs,
        post_wrappers={"RolloutInfoWrapper": lambda env, i: wrappers.RolloutInfoWrapper(env)},
    )
    try:
        yield venv
    finally:
        venv.close()


@environment_ingredient.named_config
def fast():
    num_vec = 2
    parallel = False  # easier to debug with everything in one process
    max_episode_steps = 5

    locals()  # quieten flake8
