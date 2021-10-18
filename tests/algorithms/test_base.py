"""Tests for imitation.algorithms.base."""

import numpy as np
import pytest
import torch as th

from imitation.algorithms import base
from imitation.data import types


def test_check_fixed_horizon(custom_logger):
    """Tests check for fixed horizon catches trajectories of varying lengths."""
    algo = base.BaseImitationAlgorithm(custom_logger=custom_logger)
    algo._check_fixed_horizon([])
    assert algo._horizon is None
    algo._check_fixed_horizon([5])
    assert algo._horizon == 5
    algo._check_fixed_horizon([5])
    algo._check_fixed_horizon([])
    algo._check_fixed_horizon([5, 5, 5])

    with pytest.raises(ValueError, match="Episodes of different length.*"):
        algo._check_fixed_horizon([4])
    with pytest.raises(ValueError, match="Episodes of different length.*"):
        algo._check_fixed_horizon([6])
    with pytest.raises(ValueError, match="Episodes of different length.*"):
        algo._check_fixed_horizon([1])
    assert algo._horizon == 5


def test_check_fixed_horizon_flag(custom_logger):
    """Tests check for fixed horizon ignores variable horizon with allow flag."""
    algo = base.BaseImitationAlgorithm(
        custom_logger=custom_logger,
        allow_variable_horizon=True,
    )
    algo._check_fixed_horizon([5])
    algo._check_fixed_horizon([42])
    algo._check_fixed_horizon([5, 42])
    assert algo._horizon is None


def _make_and_iterate_loader(*args, **kwargs):
    loader = base.make_data_loader(*args, **kwargs)
    for batch in loader:
        pass


def test_make_data_loader_batch_size():
    """Tests data loader performs batch size validation."""
    for batch_size in [0, -1, -42]:
        with pytest.raises(ValueError, match=".*must be positive"):
            base.make_data_loader([], batch_size=batch_size)

    # batch size = 5
    batch_iterable = [{"obs": np.zeros((5, 2)), "acts": np.zeros((5, 1))}]
    for wrong_batch_size in [4, 6, 42]:
        with pytest.raises(ValueError, match="Expected batch size.*"):
            _make_and_iterate_loader(batch_iterable, batch_size=wrong_batch_size)
    _make_and_iterate_loader(batch_iterable, batch_size=5)

    batch_iterable2 = [{"obs": np.zeros((5, 2)), "acts": np.zeros((4, 1))}]
    with pytest.raises(ValueError, match="Expected batch size.*"):
        _make_and_iterate_loader(batch_iterable2, batch_size=5)

    batch_iterable3 = [
        {"obs": np.zeros((5, 2)), "acts": np.zeros((5, 1))},
        {"obs": np.zeros((6, 2)), "acts": np.zeros((5, 1))},
    ]
    with pytest.raises(ValueError, match="Expected batch size.*"):
        _make_and_iterate_loader(batch_iterable3, batch_size=5)

    trans = types.TransitionsMinimal(
        obs=np.zeros((5, 2)),
        acts=np.zeros((5, 1)),
        infos=np.array([{}] * 5),
    )
    for smaller_bs in range(1, 6):
        base.make_data_loader(trans, batch_size=smaller_bs)
    for larger_bs in [6, 7, 42]:
        with pytest.raises(ValueError, match=".* smaller than batch size.*"):
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
            obs=np.array([10, 11, 12, 13]),
            acts=np.array([104, 105, 106]),
            infos=None,
            terminal=False,
        ),
    ]
    trans = types.Transitions(
        obs=np.array([0, 4, 5, 10, 11, 12]),
        acts=np.array([100, 102, 103, 104, 105, 106]),
        next_obs=np.array([1, 5, 6, 11, 12, 13]),
        dones=np.array([True, False, True, False, False, False]),
        infos=np.array([{}] * 6),
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
            "obs": np.array([11, 12]),
            "acts": np.array([105, 106]),
            "next_obs": np.array([12, 13]),
            "dones": np.array([False]),
            "infos": np.array([{}, {}]),
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
