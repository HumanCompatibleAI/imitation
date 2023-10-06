"""Fixtures common across tests."""
from typing import Dict

import gymnasium as gym
import numpy as np
import pytest
import seals  # noqa: F401
import torch
from stable_baselines3.common.vec_env import VecEnv

from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger, util

CARTPOLE_ENV_NAME = "seals/CartPole-v0"


@pytest.fixture(params=[1, 4], ids=lambda n: f"vecenv({n})")
def cartpole_venv(request, rng) -> VecEnv:
    num_envs = request.param
    return util.make_vec_env(
        CARTPOLE_ENV_NAME,
        n_envs=num_envs,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        rng=rng,
    )


@pytest.fixture(scope="session", autouse=True)
def torch_single_threaded():
    """Make PyTorch execute code single-threaded.

    This allows us to run the test suite with greater across-test parallelism.
    This is faster, since:
        - There are diminishing returns to more threads within a test.
        - Many tests cannot be multi-threaded (e.g. most not using PyTorch training),
          and we have to set between-test parallelism based on peak resource
          consumption of tests to avoid spurious failures.
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


@pytest.fixture()
def custom_logger(tmpdir: str) -> logger.HierarchicalLogger:
    return logger.configure(tmpdir)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=0)


@pytest.fixture()
def check_obs_or_space_equal(got, expected):
    assert type(got) is type(expected)
    if isinstance(got, (Dict, gym.spaces.Dict)):
        assert len(got.keys()) == len(expected.keys())
        for k, v in got.items():
            assert v == expected[k]
    else:
        assert got == expected
