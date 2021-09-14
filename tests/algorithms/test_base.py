"""Tests for imitation.algorithms.base."""

from typing import Sequence

import numpy as np
import pytest
import torch as th

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


def test_make_data_loader_batch_size():
    """Tests data loader performs batch size validation."""
    for batch_size in [0, -1, -42]:
        with pytest.raises(ValueError, match=".*must be positive"):
            base.make_data_loader([], batch_size=batch_size)

    # batch size = 5
    batch_iterable = [{"obs": np.zeros((5, 2)), "acts": np.zeros((5, 1))}]
    for wrong_batch_size in [4, 6, 42]:
        with pytest.raises(ValueError, match="Expected batch size.*"):
            base.make_data_loader(batch_iterable, batch_size=wrong_batch_size)
    base.make_data_loader(batch_iterable, batch_size=5)

    trans = types.TransitionsMinimal(
        obs=np.zeros((5, 2)),
        acts=np.zeros((5, 1)),
        infos=np.array([{}] * 5),
    )
    for smaller_bs in range(1, 6):
        base.make_data_loader(trans, batch_size=smaller_bs)
    for larger_bs in [6, 7, 42]:
        with pytest.raises(ValueError, match=".* < batch_size"):
            base.make_data_loader(trans, batch_size=larger_bs)


def test_make_data_loader():
    """Tests data loader produces same results for same input in different formats."""
    trajs = [
        types.Trajectory(
            obs=np.array([0, 1]),
            acts=np.array([100]),
            infos=None,
            terminal=True,
        ),
        types.Trajectory(
            obs=np.array([4, 5, 6]),
            acts=np.array([102, 103]),
            infos=None,
            terminal=True,
        ),
        types.Trajectory(
            obs=np.array([10, 11, 12]),
            acts=np.array([104, 105]),
            infos=None,
            terminal=False,
        ),
    ]
    trans = types.Transitions(
        obs=np.array([0, 4, 5, 10, 11]),
        acts=np.array([100, 102, 103, 104, 105]),
        next_obs=np.array([1, 5, 6, 11, 12]),
        dones=np.array([True, False, True, False, False]),
        infos=np.array([{}] * 5),
    )
    trans_mapping = [
        {
            "obs": np.array([0, 4]),
            "acts": np.array([100, 102]),
            "next_obs": np.array([1, 5]),
            "dones": np.array([True, False]),
            "infos": np.array([{}, {}]),
        },
        {
            "obs": np.array([5, 10]),
            "acts": np.array([103, 104]),
            "next_obs": np.array([6, 11]),
            "dones": np.array([True, False]),
            "infos": np.array([{}, {}]),
        },
        {
            "obs": np.array([11]),
            "acts": np.array([105]),
            "next_obs": np.array([12]),
            "dones": np.array([False]),
            "infos": np.array([{}]),
        },
    ]

    for data in [trajs, trans, trans_mapping]:
        data_loader = base.make_data_loader(
            data,
            batch_size=2,
            data_loader_kwargs=dict(shuffle=False, drop_last=False),
        )
        for batch, expected_batch in zip(data_loader, trans_mapping):
            assert batch.keys() == expected_batch.keys()
            for k in batch.keys():
                v = batch[k]
                if isinstance(v, th.Tensor):
                    v = v.numpy()
                assert np.all(v == expected_batch[k])
