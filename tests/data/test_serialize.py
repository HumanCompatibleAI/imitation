"""Tests for `imitation.data.serialize`."""

import pathlib

import gymnasium as gym
import numpy as np
import pytest

from imitation.data import serialize, types
from imitation.data.types import DictObs


@pytest.fixture
def data_path(tmp_path):
    return tmp_path / "data"


@pytest.mark.parametrize("path_type", [str, pathlib.Path])
def test_save_trajectory(data_path, trajectory, path_type):
    if isinstance(trajectory.obs, DictObs):
        pytest.skip("serialize.save does not yet support DictObs")

    serialize.save(path_type(data_path), [trajectory])
    assert data_path.exists()


@pytest.mark.parametrize("path_type", [str, pathlib.Path])
def test_save_trajectory_rew(data_path, trajectory_rew, path_type):
    if isinstance(trajectory_rew.obs, DictObs):
        pytest.skip("serialize.save does not yet support DictObs")
    serialize.save(path_type(data_path), [trajectory_rew])
    assert data_path.exists()


def test_save_load_trajectory(data_path, trajectory):
    if isinstance(trajectory.obs, DictObs):
        pytest.skip("serialize.save does not yet support DictObs")
    serialize.save(data_path, [trajectory])

    reconstructed = list(serialize.load(data_path))
    reconstructedi = reconstructed[0]

    assert len(reconstructed) == 1
    assert np.allclose(reconstructedi.obs, trajectory.obs)
    assert np.allclose(reconstructedi.acts, trajectory.acts)
    assert np.allclose(reconstructedi.terminal, trajectory.terminal)
    assert not hasattr(reconstructedi, "rews")


@pytest.mark.parametrize("load_fn", [serialize.load, serialize.load_with_rewards])
def test_save_load_trajectory_rew(data_path, trajectory_rew, load_fn):
    if isinstance(trajectory_rew.obs, DictObs):
        pytest.skip("serialize.save does not yet support DictObs")
    serialize.save(data_path, [trajectory_rew])

    reconstructed = list(load_fn(data_path))
    reconstructedi = reconstructed[0]

    assert len(reconstructed) == 1
    assert np.allclose(reconstructedi.obs, trajectory_rew.obs)
    assert np.allclose(reconstructedi.acts, trajectory_rew.acts)
    assert np.allclose(reconstructedi.terminal, trajectory_rew.terminal)
    assert np.allclose(reconstructedi.rews, trajectory_rew.rews)
