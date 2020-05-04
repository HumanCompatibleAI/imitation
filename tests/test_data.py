"""Tests of imitation.util.data.

Mostly checks input validation."""

import dataclasses
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
INVALID_LENGTHS = [1, 2, 10]
VALID_LENGTHS = [0] + INVALID_LENGTHS


def _check_1d_shape(fn: Callable[[np.ndarray], Any], length: float, expected_msg: str):
    for shape in [(), (length, 1), (length, 2), (length - 1,), (length + 1,)]:
        with pytest.raises(ValueError, match=expected_msg):
            fn(np.zeros(shape))


@pytest.fixture
def trajectory(
    obs_space: gym.Space, act_space: gym.Space, length: int
) -> data.Trajectory:
    """Fixture to generate trajectory of length `length` iid sampled from spaces."""
    obs = np.array([obs_space.sample() for _ in range(length + 1)])
    acts = np.array([act_space.sample() for _ in range(length)])
    infos = [{} for _ in range(length)]
    return data.Trajectory(obs=obs, acts=acts, infos=infos)


@pytest.fixture
def trajectory_rew(trajectory: data.Trajectory) -> data.TrajectoryWithRew:
    """Like `trajectory` but with reward randomly sampled from a Gaussian."""
    rews = np.random.randn(len(trajectory))
    return data.TrajectoryWithRew(**dataclasses.asdict(trajectory), rews=rews)


@pytest.fixture
def transitions(
    obs_space: gym.Space, act_space: gym.Space, length: int
) -> data.Transitions:
    """Fixture to generate transitions of length `length` iid sampled from spaces."""
    obs = np.array([obs_space.sample() for _ in range(length)])
    next_obs = np.array([obs_space.sample() for _ in range(length)])
    acts = np.array([act_space.sample() for _ in range(length)])
    dones = np.zeros(length, dtype=np.bool)
    return data.Transitions(obs=obs, acts=acts, next_obs=next_obs, dones=dones)


@pytest.fixture
def transitions_rew(transitions: data.Transitions) -> data.TransitionsWithRew:
    """Like `transitions` but with reward randomly sampled from a Gaussian."""
    rews = np.random.randn(len(transitions))
    return data.TransitionsWithRew(**dataclasses.asdict(transitions), rews=rews)


@pytest.mark.parametrize("obs_space", OBS_SPACES)
@pytest.mark.parametrize("act_space", ACT_SPACES)
class TestData:
    """Tests of imitation.util.data.

    Grouped in a class since parametrized over common set of spaces.
    """

    @pytest.mark.parametrize("length", VALID_LENGTHS)
    def test_valid_trajectories(
        self,
        trajectory: data.Trajectory,
        trajectory_rew: data.TrajectoryWithRew,
        length: int,
    ) -> None:
        """Checks trajectories can be created for a variety of lengths and spaces."""
        trajs = [trajectory, trajectory_rew]
        trajs += [dataclasses.replace(traj, infos=None) for traj in trajs]
        for traj in trajs:
            assert len(traj) == length

    @pytest.mark.parametrize("length", INVALID_LENGTHS)
    def test_invalid_trajectories(
        self, trajectory: data.Trajectory, trajectory_rew: data.TrajectoryWithRew,
    ) -> None:
        """Checks input validation catches space and dtype related errors."""
        trajs = [trajectory, trajectory_rew]
        for traj in trajs:
            with pytest.raises(
                ValueError, match=r"expected one more observations than actions.*"
            ):
                dataclasses.replace(traj, obs=traj.obs[:-1])
            with pytest.raises(
                ValueError, match=r"expected one more observations than actions.*"
            ):
                dataclasses.replace(traj, acts=traj.acts[:-1])

            with pytest.raises(
                ValueError,
                match=r"infos when present must be present for each action.*",
            ):
                dataclasses.replace(traj, infos=traj.infos[:-1])
            with pytest.raises(
                ValueError,
                match=r"infos when present must be present for each action.*",
            ):
                dataclasses.replace(traj, obs=traj.obs[:-1], acts=traj.acts[:-1])

        _check_1d_shape(
            fn=lambda rews: dataclasses.replace(trajectory_rew, rews=rews),
            length=len(trajectory_rew),
            expected_msg=r"rewards must be 1D array.*",
        )

        with pytest.raises(ValueError, match=r"rewards dtype.* not a float"):
            dataclasses.replace(
                trajectory_rew, rews=np.zeros(len(trajectory_rew), dtype=np.int)
            )

    @pytest.mark.parametrize("length", VALID_LENGTHS)
    def test_valid_transitions(
        self,
        transitions: data.Transitions,
        transitions_rew: data.TransitionsWithRew,
        length: int,
    ) -> None:
        """Checks trajectories can be created for a variety of lengths and spaces."""
        assert len(transitions) == length
        assert len(transitions_rew) == length

    @pytest.mark.parametrize("length", INVALID_LENGTHS)
    def test_invalid_transitions(
        self, transitions: data.Transitions, transitions_rew: data.TransitionsWithRew,
    ) -> None:
        """Checks input validation catches space and dtype related errors."""
        for trans in [transitions, transitions_rew]:
            with pytest.raises(
                ValueError, match=r"obs and next_obs must have same shape:.*"
            ):
                dataclasses.replace(trans, obs=trans.obs[:-1])
            with pytest.raises(
                ValueError, match=r"obs and next_obs must have same shape:.*"
            ):
                dataclasses.replace(trans, next_obs=np.zeros((len(trans), 4, 2)))

            with pytest.raises(
                ValueError, match=r"obs and next_obs must have the same dtype:.*"
            ):
                dataclasses.replace(
                    trans, next_obs=np.zeros_like(trans.obs, dtype=np.bool)
                ),

            with pytest.raises(
                ValueError, match=r"obs and acts must have same number of timesteps:.*"
            ):
                dataclasses.replace(
                    trans, obs=trans.obs[:-1], next_obs=trans.next_obs[:-1]
                )
            with pytest.raises(
                ValueError, match=r"obs and acts must have same number of timesteps:.*"
            ):
                dataclasses.replace(trans, acts=trans.acts[:-1])

            _check_1d_shape(
                fn=lambda bogus_dones: dataclasses.replace(trans, dones=bogus_dones),
                length=len(trans),
                expected_msg=r"dones must be 1D array.*",
            )

            with pytest.raises(ValueError, match=r"dones must be boolean"):
                dataclasses.replace(trans, dones=np.zeros(len(trans), dtype=np.int))

        _check_1d_shape(
            fn=lambda bogus_rews: dataclasses.replace(trans, rews=bogus_rews),
            length=len(transitions_rew),
            expected_msg=r"rewards must be 1D array.*",
        )

        with pytest.raises(ValueError, match=r"rewards dtype.* not a float"):
            dataclasses.replace(
                transitions_rew, rews=np.zeros(len(transitions_rew), dtype=np.int)
            )
