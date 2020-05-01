"""Tests of imitation.util.data.

Mostly checks input validation."""

import functools
from typing import Any, Callable

import gym
import numpy as np
import pytest

from imitation.util import data

SPACES = [
    gym.spaces.Discrete(3),
    gym.spaces.MultiDiscrete((3, 4)),
    gym.spaces.Box(-1, 1, shape=(1,)),
    gym.spaces.Box(-1, 1, shape=(2,)),
    gym.spaces.Box(-np.inf, np.inf, shape=(2,)),
]
OBS_SPACES = SPACES
ACT_SPACES = SPACES


def _check_1d_shape(fn: Callable[[np.ndarray], Any], length: float, expected_msg: str):
    for shape in [(), (length, 1), (length, 2), (length - 1,), (length + 1,)]:
        with pytest.raises(ValueError, match=expected_msg):
            fn(np.zeros(shape))


@pytest.mark.parametrize("obs_space", OBS_SPACES)
@pytest.mark.parametrize("act_space", ACT_SPACES)
@pytest.mark.parametrize("length", [0, 1, 2, 10])
def test_valid_trajectories(
    obs_space: gym.Space, act_space: gym.Space, length: int
) -> None:
    """Checks trajectories can be created for a variety of lengths and spaces."""
    obs = np.array([obs_space.sample() for _ in range(length + 1)])
    acts = np.array([act_space.sample() for _ in range(length)])
    infos = [{} for _ in range(length)]
    rews = np.random.randn(length)

    traja = data.Trajectory(obs=obs, acts=acts, infos=None)
    trajb = data.Trajectory(obs=obs, acts=acts, infos=infos)
    trajc = data.TrajectoryWithRew(obs=obs, acts=acts, infos=None, rews=rews)
    trajd = data.TrajectoryWithRew(obs=obs, acts=acts, infos=infos, rews=rews)
    for traj in [traja, trajb, trajc, trajd]:
        assert len(traj) == length


@pytest.mark.parametrize("obs_space", OBS_SPACES)
@pytest.mark.parametrize("act_space", ACT_SPACES)
@pytest.mark.parametrize("length", [1, 2, 10])
def test_invalid_trajectories(
    obs_space: gym.Space, act_space: gym.Space, length: int
) -> None:
    """Checks input validation catches space and dtype related errors."""
    obs = np.array([obs_space.sample() for _ in range(length + 1)])
    acts = np.array([act_space.sample() for _ in range(length)])
    infos = [{} for _ in range(length)]
    rews = np.random.randn(length)

    for cls in [data.Trajectory, functools.partial(data.TrajectoryWithRew, rews=rews)]:
        with pytest.raises(
            ValueError, match=r"expected one more observations than actions.*"
        ):
            cls(obs=obs[:-1], acts=acts, infos=None)
        with pytest.raises(
            ValueError, match=r"expected one more observations than actions.*"
        ):
            cls(obs=obs, acts=acts[:-1], infos=None)

        with pytest.raises(
            ValueError, match=r"infos when present must be present for each action.*"
        ):
            cls(obs=obs, acts=acts, infos=infos[:-1])
        with pytest.raises(
            ValueError, match=r"infos when present must be present for each action.*"
        ):
            cls(obs=obs[:-1], acts=acts[:-1], infos=infos)

    _check_1d_shape(
        fn=lambda rews: data.TrajectoryWithRew(
            obs=obs, acts=acts, infos=infos, rews=rews
        ),
        length=length,
        expected_msg=r"rewards must be 1D array.*",
    )

    with pytest.raises(ValueError, match=r"rewards dtype.* not a float"):
        data.TrajectoryWithRew(
            obs=obs, acts=acts, infos=infos, rews=np.zeros(length, dtype=np.int)
        )


@pytest.mark.parametrize("obs_space", OBS_SPACES)
@pytest.mark.parametrize("act_space", ACT_SPACES)
@pytest.mark.parametrize("length", [0, 1, 2, 10])
def test_valid_transitions(
    obs_space: gym.Space, act_space: gym.Space, length: int
) -> None:
    """Checks trajectories can be created for a variety of lengths and spaces."""
    obs = np.array([obs_space.sample() for _ in range(length)])
    next_obs = np.array([obs_space.sample() for _ in range(length)])
    acts = np.array([act_space.sample() for _ in range(length)])
    dones = np.zeros(length, dtype=np.bool)
    rews = np.random.randn(length)

    trans = data.Transitions(obs=obs, acts=acts, next_obs=next_obs, dones=dones)
    trans_rew = data.TransitionsWithRew(
        obs=obs, acts=acts, next_obs=next_obs, dones=dones, rews=rews
    )
    for traj in [trans, trans_rew]:
        assert len(traj) == length


@pytest.mark.parametrize("obs_space", OBS_SPACES)
@pytest.mark.parametrize("act_space", ACT_SPACES)
@pytest.mark.parametrize("length", [1, 2, 10])
def test_invalid_transitions(
    obs_space: gym.Space, act_space: gym.Space, length: int
) -> None:
    """Checks input validation catches space and dtype related errors."""
    obs = np.array([obs_space.sample() for _ in range(length)])
    next_obs = np.array([obs_space.sample() for _ in range(length)])
    acts = np.array([act_space.sample() for _ in range(length)])
    dones = np.zeros(length, dtype=np.bool)
    rews = np.random.randn(length)

    for cls in [
        data.Transitions,
        functools.partial(data.TransitionsWithRew, rews=rews),
    ]:
        with pytest.raises(
            ValueError, match=r"obs and next_obs must have same shape:.*"
        ):
            cls(obs=obs[:-1], acts=acts, next_obs=next_obs, dones=dones)
        with pytest.raises(
            ValueError, match=r"obs and next_obs must have same shape:.*"
        ):
            cls(obs=obs, acts=acts, next_obs=np.zeros((length, 4, 2)), dones=dones)

        with pytest.raises(
            ValueError, match=r"obs and next_obs must have the same dtype:.*"
        ):
            cls(
                obs=obs,
                acts=acts,
                next_obs=np.zeros_like(obs, dtype=np.bool),
                dones=dones,
            )

        with pytest.raises(
            ValueError, match=r"obs and acts must have same number of timesteps:.*"
        ):
            cls(obs=obs[:-1], acts=acts, next_obs=next_obs[:-1], dones=dones)
        with pytest.raises(
            ValueError, match=r"obs and acts must have same number of timesteps:.*"
        ):
            cls(obs=obs, acts=acts[:-1], next_obs=next_obs, dones=dones)

        _check_1d_shape(
            fn=lambda bogus_dones: cls(
                obs=obs, acts=acts, next_obs=next_obs, dones=bogus_dones,
            ),
            length=length,
            expected_msg=r"dones must be 1D array.*",
        )

        with pytest.raises(ValueError, match=r"dones must be boolean"):
            cls(
                obs=obs,
                acts=acts,
                next_obs=next_obs,
                dones=np.zeros(length, dtype=np.int),
            )

    _check_1d_shape(
        fn=lambda bogus_rews: data.TransitionsWithRew(
            obs=obs, acts=acts, next_obs=next_obs, dones=dones, rews=bogus_rews
        ),
        length=length,
        expected_msg=r"rewards must be 1D array.*",
    )

    with pytest.raises(ValueError, match=r"rewards dtype.* not a float"):
        data.TransitionsWithRew(
            obs=obs,
            acts=acts,
            next_obs=next_obs,
            dones=dones,
            rews=np.zeros(length, dtype=np.int),
        )
