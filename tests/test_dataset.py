# import numpy as np
# import pytest
#
# from imitation.data import types
# import imitation.data.dataset as ds
#
#
# @pytest.fixture(params=(ds.EpochOrderSimpleDataset, ds.RandomSimpleDataset))
# def simple_dataset_cls(request) -> ds.SimpleDataset:
#     return request.config
#
# TRANSITION_SHAPES = (
#     {"obs": (}
# )
#
#
# TRANSITIONS = (
#     types.Transitions()
# )
#
#
# @pytest.fixture
# def transitions():
#     pass
#
#
# @pytest.fixture(
#     params=()
# )
# def data_map(request):
#     request.config
#
#
# @pytest.fixture
# def simple_dataset(simple_dataset_cls, data_map):
#     return simple_dataset_cls(data_map)
#
#
# def test_simple_dataset_len(simple_dataset, data_map):
#     pass
#
#
# def test_simple_dataset_copy_arrays(simple_dataset_cls):
#     """Checks that internal data_map isn't modified by external array write."""
#     data_map = {"ones": np.ones(5,)}
#     simple_dataset: ds.SimpleDataset = simple_dataset_cls(data_map)
#     def check():
#         assert set(simple_dataset.data_map.keys()) == {"ones"}
#         assert np.all(simple_dataset.data_map.sample(10) == 1)
#     check()
#     data_map["ones"][0] = 0
#     data_map["zeros"] = np.zeros(5,)
#     check()
#
