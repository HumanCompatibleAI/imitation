"""Tests of `imitation.data.{dataset,types}`."""


import collections
import copy
import dataclasses
from typing import Any, Callable

import gym
import numpy as np
import numpy.testing as npt
import pytest

from imitation.data import datasets, types

SPACES = [
    gym.spaces.Discrete(3),
    gym.spaces.MultiDiscrete((3, 4)),
    gym.spaces.Box(-1, 1, shape=(1,)),
    gym.spaces.Box(-1, 1, shape=(2,)),
    gym.spaces.Box(-np.inf, np.inf, shape=(2,)),
]
OBS_SPACES = SPACES
ACT_SPACES = SPACES
LENGTHS = [1, 2, 10]


def _check_1d_shape(fn: Callable[[np.ndarray], Any], length: float, expected_msg: str):
    for shape in [(), (length, 1), (length, 2), (length - 1,), (length + 1,)]:
        with pytest.raises(ValueError, match=expected_msg):
            fn(np.zeros(shape))


@pytest.fixture
def trajectory(
    obs_space: gym.Space, act_space: gym.Space, length: int
) -> types.Trajectory:
    """Fixture to generate trajectory of length `length` iid sampled from spaces."""
    obs = np.array([obs_space.sample() for _ in range(length + 1)])
    acts = np.array([act_space.sample() for _ in range(length)])
    infos = np.array([{} for _ in range(length)])
    return types.Trajectory(obs=obs, acts=acts, infos=infos)


@pytest.fixture
def trajectory_rew(trajectory: types.Trajectory) -> types.TrajectoryWithRew:
    """Like `trajectory` but with reward randomly sampled from a Gaussian."""
    rews = np.random.randn(len(trajectory))
    return types.TrajectoryWithRew(**dataclasses.asdict(trajectory), rews=rews)


@pytest.fixture
def transitions_min(
    obs_space: gym.Space, act_space: gym.Space, length: int
) -> types.TransitionsMinimal:
    obs = np.array([obs_space.sample() for _ in range(length)])
    acts = np.array([act_space.sample() for _ in range(length)])
    infos = np.array([{}] * length)
    return types.TransitionsMinimal(obs=obs, acts=acts, infos=infos)


@pytest.fixture
def transitions(
    transitions_min: types.TransitionsMinimal, obs_space: gym.Space, length: int
) -> types.Transitions:
    """Fixture to generate transitions of length `length` iid sampled from spaces."""
    next_obs = np.array([obs_space.sample() for _ in range(length)])
    dones = np.zeros(length, dtype=np.bool)
    return types.Transitions(
        **dataclasses.asdict(transitions_min), next_obs=next_obs, dones=dones
    )


@pytest.fixture
def transitions_rew(
    transitions: types.Transitions, length: int
) -> types.TransitionsWithRew:
    """Like `transitions` but with reward randomly sampled from a Gaussian."""
    rews = np.random.randn(length)
    return types.TransitionsWithRew(**dataclasses.asdict(transitions), rews=rews)


@pytest.mark.parametrize("obs_space", OBS_SPACES)
@pytest.mark.parametrize("act_space", ACT_SPACES)
@pytest.mark.parametrize("length", LENGTHS)
class TestData:
    """Tests of imitation.util.data.

    Grouped in a class since parametrized over common set of spaces.
    """

    def test_valid_trajectories(
        self,
        trajectory: types.Trajectory,
        trajectory_rew: types.TrajectoryWithRew,
        length: int,
    ) -> None:
        """Checks trajectories can be created for a variety of lengths and spaces."""
        trajs = [trajectory, trajectory_rew]
        trajs += [dataclasses.replace(traj, infos=None) for traj in trajs]
        for traj in trajs:
            assert len(traj) == length

    def test_invalid_trajectories(
        self, trajectory: types.Trajectory, trajectory_rew: types.TrajectoryWithRew,
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

    def test_valid_transitions(
        self,
        transitions_min: types.TransitionsMinimal,
        transitions: types.Transitions,
        transitions_rew: types.TransitionsWithRew,
        length: int,
    ) -> None:
        """Checks trajectories can be created for a variety of lengths and spaces."""
        assert len(transitions_min) == length
        assert len(transitions) == length
        assert len(transitions_rew) == length

    def test_invalid_transitions(
        self,
        transitions_min: types.Transitions,
        transitions: types.Transitions,
        transitions_rew: types.TransitionsWithRew,
        length: int,
    ) -> None:

        for trans in [transitions_min, transitions, transitions_rew]:
            with pytest.raises(
                ValueError, match=r"obs and acts must have same number of timesteps:.*"
            ):
                dataclasses.replace(trans, acts=trans.acts[:-1])
            with pytest.raises(
                ValueError, match=r"obs and infos must have same number of timesteps:.*"
            ):
                dataclasses.replace(trans, infos=[{}] * (length - 1))

        """Checks input validation catches space and dtype related errors."""
        for trans in [transitions, transitions_rew]:
            with pytest.raises(
                ValueError, match=r"obs and next_obs must have same shape:.*"
            ):
                dataclasses.replace(trans, next_obs=np.zeros((len(trans), 4, 2)))

            with pytest.raises(
                ValueError, match=r"obs and next_obs must have the same dtype:.*"
            ):
                dataclasses.replace(
                    trans, next_obs=np.zeros_like(trans.next_obs, dtype=np.bool)
                ),

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


def test_zero_length_fails():
    """Check zero-length trajectory and transitions fail."""
    empty = np.array([])
    with pytest.raises(ValueError, match=r"Degenerate trajectory.*"):
        types.Trajectory(obs=np.array([42]), acts=empty, infos=None)
    with pytest.raises(ValueError, match=r"Must have non-zero number of.*"):
        types.Transitions(
            obs=empty,
            acts=empty,
            next_obs=empty,
            dones=empty.astype(np.bool),
            infos=empty.astype(np.object),
        )


# Dataset tests:
# (In same file for now because Transition fixture are handy).


DICT_DATASET_PARAMS = [
    (datasets.EpochOrderDictDataset, {"shuffle": False}),
    (datasets.EpochOrderDictDataset, {"shuffle": True}),
    (datasets.RandomDictDataset, {}),
]

DATA_MAP = [
    {"a": np.zeros([10, 2], dtype=int), "b": np.random.standard_normal([10])},
    {"asdf": np.zeros([100, 2, 4])},
    {"asdf": np.zeros([97, 2, 4]), "foo": np.ones([97, 1], dtype=bool)},
]


@pytest.fixture(params=DICT_DATASET_PARAMS)
def dict_dataset_params(request):
    return request.param


@pytest.fixture(params=DATA_MAP)
def data_map(request):
    return copy.deepcopy(request.param)


@pytest.fixture
def dict_dataset(dict_dataset_params, data_map):
    dataset_cls, kwargs = dict_dataset_params
    return dataset_cls(data_map, **kwargs)


def test_dict_dataset_copy_data_map(dict_dataset: datasets.DictDataset, data_map):
    """Check that `dict_dataset.data_map` is unchanged by writes to `data_map` param.
    Note that the `data_map` fixture supplies the same instance `data_map` that was used
    to initialize the `dict_dataset` fixture.
    """
    backup_map = copy.deepcopy(dict_dataset.data_map)
    npt.assert_equal(backup_map, dict_dataset.data_map)
    for v in data_map.values():
        # Modify array in-place.
        np.add(v, v.dtype.type(3), out=v)
        np.multiply(v, v.dtype.type(2), out=v)
    npt.assert_equal(backup_map, dict_dataset.data_map)


def test_dict_dataset_dtypes(dict_dataset, data_map, max_batch_size=80, n_checks=20):
    """Check that DictDataset preserves array dtype."""
    for _ in range(n_checks):
        n_samples = np.random.randint(max_batch_size) + 1
        sample = dict_dataset.sample(n_samples)
        for k in data_map.keys():
            assert data_map[k].dtype == sample[k].dtype


@pytest.mark.parametrize("n_samples", [-i for i in range(4)])
def test_dict_dataset_nonpositive_samples_error(dict_dataset, n_samples):
    """Check that DictDataset errors on n_samples<=0"""
    with pytest.raises(ValueError, match="n_samples"):
        dict_dataset.sample(n_samples)


def test_dict_dataset_data_map_error(dict_dataset_params):
    """Check that DictDataset errors on unequal number of data_map rows."""
    dataset_cls, kwargs = dict_dataset_params
    with pytest.raises(ValueError, match="Empty.*"):
        dataset_cls({}, **kwargs)

    unequal_data_map = {
        "a": np.zeros([10]),
        "b": np.zeros([10, 1]),
        "c": np.zeros([10, 2]),
        "d": np.zeros([2, 10]),
    }
    with pytest.raises(ValueError, match="Unequal.*"):
        dataset_cls(unequal_data_map, **kwargs)


def test_dict_dataset_parallel_rows(
    dict_dataset_params, max_batch_size=80, n_checks=20
):
    """Check that 'parallel rows' among different key-value pairs remain parallel.

    Nontrivially, shuffled datasets should maintain this order.
    """
    dataset_cls, kwargs = dict_dataset_params
    range_data_map = {k: i + np.arange(50,) for i, k in enumerate("abcd")}
    dict_dataset = dataset_cls(range_data_map, **kwargs)
    for _ in range(n_checks):
        n_samples = np.random.randint(max_batch_size) + 1
        sample = dict_dataset.sample(n_samples)
        for i, k in enumerate("abcd"):
            np.testing.assert_array_equal(sample[k], i + sample["a"])


def test_dict_dataset_correct_sample_len(dict_dataset, max_batch_size=80):
    """Check that DictDataset returns samples of the right length."""
    batch_sizes = np.arange(max_batch_size) + 1
    np.random.shuffle(batch_sizes)
    for n_samples in batch_sizes:
        sample = dict_dataset.sample(n_samples)
        for v in sample.values():
            assert len(v) == n_samples


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("dataset_size", [1, 30, 100, 200])
class TestEpochOrderDictDataset:
    @pytest.fixture
    def arange_dataset(self, shuffle, dataset_size):
        data_map = {"a": np.arange(dataset_size)}
        ds = datasets.EpochOrderDictDataset(data_map, shuffle=shuffle)
        return ds

    def test_epoch_order_dict_dataset_shuffle_order(
        self, arange_dataset, shuffle, dataset_size, n_checks=3,
    ):
        """Check that epoch order is deterministic iff not shuffled.

        The check has `1 / factorial(dataset_size)` chance of false negative when
        shuffle is True, so we skip on smaller dataset_sizes.
        """
        if dataset_size < 20 and shuffle:
            pytest.skip("False negative chance too high.")
        for _ in range(n_checks):
            first = arange_dataset.sample(dataset_size)
            second = arange_dataset.sample(dataset_size)
            assert len(first.keys()) == len(second.keys())
            same_order = np.all(np.equal(first[k], second[k]) for k in first.keys())
            assert same_order != shuffle

    def test_epoch_order_dict_dataset_order_property(
        self, arange_dataset, max_batch_size=31, n_epochs=4,
    ):
        """No sample should be returned n+1 times until others are returned n times."""
        counter = collections.Counter({i: 0 for i in range(arange_dataset.size())})
        n_samples_total = 0
        for epoch_num in range(n_epochs):
            while n_samples_total < (epoch_num + 1) * arange_dataset.size():
                n_samples = np.random.randint(max_batch_size) + 1
                sample = arange_dataset.sample(n_samples)
                n_samples_total += n_samples
                counter.update(list(sample["a"]))
                counts = set(counter.values())
                if len(counts) == 1:
                    # Only happens if on epoch boundary.
                    assert n_samples_total % arange_dataset.size() == 0
                else:
                    assert len(counts) == 2
                    assert min(counts) == max(counts) - 1


class TestTransitionsDictDatasetAdaptor(TestData):
    # Subclassing TestData gives access to parametrized `transitions.*` fixtures
    # because this class shares `pytest.mark.parametrized` with superclass.

    @pytest.fixture
    def trans_ds(self, transitions, dict_dataset_params):
        dict_dataset_cls, dict_dataset_kwargs = dict_dataset_params
        return datasets.TransitionsDictDatasetAdaptor(
            transitions, dict_dataset_cls, dict_dataset_kwargs
        )

    @pytest.fixture
    def trans_ds_rew(self, transitions_rew, dict_dataset_params):
        dict_dataset_cls, dict_dataset_kwargs = dict_dataset_params
        return datasets.TransitionsDictDatasetAdaptor(
            transitions_rew, dict_dataset_cls, dict_dataset_kwargs
        )

    def test_size(self, trans_ds, transitions, trans_ds_rew, transitions_rew):
        """Check for the correct size()."""
        assert len(transitions) == len(transitions_rew)  # Sanity check...
        assert trans_ds.size() == trans_ds_rew.size() == len(transitions)

    def test_sample_sizes_and_types(
        self,
        trans_ds,
        transitions,
        trans_ds_rew,
        transitions_rew,
        max_batch_size=50,
        n_checks=30,
    ):
        """Check for correct sample shapes and dtypes."""
        for ds, trans in [(trans_ds, transitions), (trans_ds_rew, transitions_rew)]:
            trans_dict = dataclasses.asdict(trans)
            for _ in range(n_checks):
                n_samples = np.random.randint(max_batch_size) + 1
                sample = ds.sample(n_samples)
                assert isinstance(sample, type(trans))
                for k, v in dataclasses.asdict(sample).items():
                    if k == "infos":
                        trans_v: np.ndarray = np.array(trans_dict[k])
                    else:
                        trans_v: np.ndarray = trans_dict[k]
                    assert v.dtype == trans_v.dtype
                    assert v.shape[1:] == trans_v.shape[1:]
                    assert v.shape[0] == n_samples
