"""Tests for imitation.algorithms.base."""

from typing import Sequence

import numpy as np
import pytest

from imitation.algorithms import base
from imitation.data import types


def gen_trajectories(
    lens: Sequence[int],
    terminal: Sequence[bool],
) -> Sequence[types.Trajectory]:
    """Generate trajectories of lengths specified in `lens`."""
    trajs = []
    for n, t in zip(lens, terminal):
        traj = types.Trajectory(
            obs=np.zeros((n + 1, 2)),
            acts=np.zeros((n,)),
            infos=None,
            terminal=t,
        )
        trajs.append(traj)
    return trajs


def test_check_fixed_horizon(custom_logger):
    """Tests check for fixed horizon catches trajectories of varying lengths."""
    algo = base.BaseImitationAlgorithm(custom_logger=custom_logger)
    algo._check_fixed_horizon(trajs=[])
    assert algo._horizon is None
    algo._check_fixed_horizon(trajs=gen_trajectories([5], [True]))
    assert algo._horizon == 5
    algo._check_fixed_horizon(trajs=gen_trajectories([5], [True]))
    algo._check_fixed_horizon(trajs=[])
    algo._check_fixed_horizon(trajs=gen_trajectories([5, 5, 5], [True, True, True]))

    with pytest.raises(ValueError, match="Episodes of different length.*"):
        algo._check_fixed_horizon(trajs=gen_trajectories([4], [True]))
    with pytest.raises(ValueError, match="Episodes of different length.*"):
        algo._check_fixed_horizon(trajs=gen_trajectories([6], [True]))
    with pytest.raises(ValueError, match="Episodes of different length.*"):
        algo._check_fixed_horizon(trajs=gen_trajectories([1], [True]))

    algo._check_fixed_horizon(trajs=gen_trajectories([4, 6, 1], [False, False, False]))
    algo._check_fixed_horizon(trajs=gen_trajectories([42], [False]))
    algo._check_fixed_horizon(trajs=gen_trajectories([5], [True]))
    assert algo._horizon == 5


def test_check_fixed_horizon_flag(custom_logger):
    """Tests check for fixed horizon ignores variable horizon with allow flag."""
    algo = base.BaseImitationAlgorithm(
        custom_logger=custom_logger,
        allow_variable_horizon=True,
    )
    algo._check_fixed_horizon(trajs=gen_trajectories([5], [True]))
    algo._check_fixed_horizon(trajs=gen_trajectories([42], [True]))
    algo._check_fixed_horizon(trajs=gen_trajectories([5, 42], [True, True]))
    assert algo._horizon is None
