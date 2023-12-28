"""Tests for imitation.data.huggingface_utils."""
import datetime
import pathlib
from typing import Sequence, cast

import hypothesis
import hypothesis.strategies as st
import numpy as np

import imitation.testing.hypothesis_strategies as h_strats
from imitation.data import huggingface_utils, serialize, types


def wrap_in_trajectory_dataset_sequence(
    trajectories: Sequence[types.Trajectory],
) -> huggingface_utils.TrajectoryDatasetSequence:
    """Takes a list of trajectories in a TrajectoryDatasetSequence.

    It first converts the trajectories to a dataset and then presents that
    dataset as a sequence of trajectories again using TrajectoryDatasetSequence.

    This is useful for testing the TrajectoryDatasetSequence class without having to
    load a dataset from the hub or disk.

    Args:
        trajectories: The trajectories to wrap.

    Returns:
        The wrapped trajectories.
    """
    return huggingface_utils.TrajectoryDatasetSequence(
        huggingface_utils.trajectories_to_dataset(trajectories),
    )


@hypothesis.given(trajectories=h_strats.trajectories_list, do_wrap=st.booleans())
@hypothesis.settings(
    suppress_health_check=[
        # Note: needed to convince hypothesis that it is ok that we use the same
        #   temporary directory for all tests.
        hypothesis.HealthCheck.function_scoped_fixture,
    ],
    deadline=datetime.timedelta(seconds=0.5),  # reduce flakiness
)
def test_save_load_roundtrip(
    trajectories: Sequence[types.Trajectory],
    do_wrap: bool,
    tmpdir: pathlib.Path,
):
    """Test saving and loading a sequence of trajectories."""
    # GIVEN
    if do_wrap:
        trajectories = wrap_in_trajectory_dataset_sequence(trajectories)

    # WHEN
    serialize.save(tmpdir, trajectories)
    loaded_trajectories = serialize.load(tmpdir)

    # THEN
    assert len(trajectories) == len(loaded_trajectories)
    for traj, loaded_traj in zip(trajectories, loaded_trajectories):
        assert traj == loaded_traj


@hypothesis.given(st.data(), h_strats.trajectories_list)
# the first run sometimes takes longer, so we give it more time
@hypothesis.settings(deadline=datetime.timedelta(milliseconds=300))
def test_sliced_access(data: st.DataObject, trajectories: Sequence[types.Trajectory]):
    """Test that slicing a TrajectoryDatasetSequence behaves as expected."""
    # GIVEN
    wrapped_trajectories = wrap_in_trajectory_dataset_sequence(trajectories)
    slices_strategy = st.slices(len(trajectories))

    # Note: we test if for 10 slices at a time because creating the dataset is slow
    for _ in range(10):
        # GIVEN
        the_slice = data.draw(slices_strategy)
        indices_of_slice = list(range(*the_slice.indices(len(trajectories))))

        # WHEN
        sliced_data = wrapped_trajectories[the_slice]

        # THEN
        for idx, traj in zip(indices_of_slice, sliced_data):
            assert traj == trajectories[idx]


@hypothesis.given(st.data(), h_strats.trajectory)
# the first run sometimes takes longer, so we give it more time
@hypothesis.settings(deadline=datetime.timedelta(milliseconds=300))
def test_sliced_info_dict_access(
    data: st.DataObject,
    trajectory: types.Trajectory,
) -> None:
    """Test slicing of info dicts from a TrajectoryDatasetSequence.

    Note: this test ensures coverage of sliced access to info dicts,
    which are wrapped in a `_LazyDecodedList`.

    Args:
        data: The hypothesis data object.
        trajectory: The trajectory to test slicing on.
    """
    # GIVEN
    # Note: we need to add the cast here since for some reason, the type of the
    # __getitem__ method is not inferred correctly.
    wrapped_trajectory = cast(
        types.Trajectory,
        wrap_in_trajectory_dataset_sequence([trajectory])[0],
    )
    # Note: the cast of the info dicts is needed since the `infos` field of a
    #  trajectory is optional.
    info_dicts = cast(np.ndarray, trajectory.infos)
    wrapped_info_dicts = cast(np.ndarray, wrapped_trajectory.infos)
    slices_strategy = st.slices(len(wrapped_info_dicts))

    # Note: we test if for 10 slices at a time because creating the dataset is slow
    for _ in range(10):
        # GIVEN
        the_slice = data.draw(slices_strategy)
        indices_of_slice = list(range(*the_slice.indices(len(wrapped_info_dicts))))

        # WHEN
        info_slice = wrapped_info_dicts[the_slice]

        # THEN
        for idx, info_dict in zip(indices_of_slice, info_slice):
            assert info_dict == info_dicts[idx]
