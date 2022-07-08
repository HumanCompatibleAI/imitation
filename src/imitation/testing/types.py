"""Helper methods for tests involving imitation.data.types."""

import dataclasses
from typing import Sequence

import numpy as np

from imitation.data import types


def assert_traj_equal(traj_a: types.Trajectory, traj_b: types.Trajectory) -> None:
    """Assert trajectory `a` and `b` are equal."""
    dict_a, dict_b = dataclasses.asdict(traj_a), dataclasses.asdict(traj_b)
    assert dict_a.keys() == dict_b.keys()
    for k, a_v in dict_a.items():
        b_v = dict_b[k]
        if k == "infos":
            # Treat None equivalent to sequence of empty dicts
            a_v = [{}] * len(traj_a) if a_v is None else a_v
            b_v = [{}] * len(traj_b) if b_v is None else b_v
        assert np.array_equal(a_v, b_v)


def assert_traj_sequences_equal(
    traj_seq_a: Sequence[types.Trajectory],
    traj_seq_b: Sequence[types.Trajectory],
) -> None:
    """Assert trajectory sequences `a` and `b` are equal."""
    assert len(traj_seq_a) == len(traj_seq_b)
    for traj_a, traj_b in zip(traj_seq_a, traj_seq_b):
        assert_traj_equal(traj_a, traj_b)
