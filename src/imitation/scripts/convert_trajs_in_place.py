"""
This script takes as command-line input multiple paths to pickle files containing
Sequence[imitation.types.TrajectoryWithRew] or Sequence[imitation.old_types.Trajectory].
It overwrites each path with a new pickle where the demonstration data is saved as
Sequence[imitation.types.TrajectoryWithRew].
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
            message=".*trajectories are saved in an outdated format.*",
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
