"""Converts old-style trajectories to new-style trajectories.

This script takes as command-line input multiple paths to pickle files containing
(possibly old forms of) Sequence[imitation.types.TrajectoryWithRew]. It overwrites each
path with a new pickle where the demonstration data is saved as
Sequence[imitation.types.TrajectoryWithRew]. This is updated to the newest format,
setting `terminal=True` where it is missing.
"""

import warnings

from imitation.data import types


def update_traj_file_in_place(path: str) -> None:
    """Modifies trajectories pickle file in-place to update data to new format.

    The new data is saved as `Sequence[imitation.types.TrajectoryWithRew]`.

    Args:
        path: Path to a pickle file containing `Sequence[imitation.types.Trajectory]`
            or `Sequence[imitation.old_types.TrajectoryWithRew]`.
    """
    with warnings.catch_warnings():
        # Filter out DeprecationWarning because we expect to load old trajectories here.
        warnings.filterwarnings(
            "ignore",
            message="Loading old version of Trajectory.*",
            category=DeprecationWarning,
        )
        trajs = types.load(path)
    types.save(path, trajs)


def main():
    import sys

    if len(sys.argv) <= 1:
        print("Supply at least one path to pickled trajectories.")
    else:
        for path in sys.argv[1:]:
            update_traj_file_in_place(path)


if __name__ == "__main__":
    main()
