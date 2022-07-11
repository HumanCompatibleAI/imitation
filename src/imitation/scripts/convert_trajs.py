"""Converts old-style pickle trajectories to new-style NPZ trajectories.

See https://github.com/HumanCompatibleAI/imitation/pull/448 for a description
of the new trajectory format.

This script takes as command-line input multiple paths to saved trajectories,
in the old .pkl or new .npz format. It then saves each sequence in the new .npz
format. The path is the same as the original but with an ".npz" extension
(i.e. "A.pkl" -> "A.npz", "A.npz" -> "A.npz", "A" -> "A.npz", "A.foo" -> "A.foo.npz").
"""

import os
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

    path, ext = os.path.splitext(path)
    new_ext = ".npz" if ext in (".pkl", ".npz") else ext + ".npz"
    types.save(path + new_ext, trajs)


def main():
    import sys

    if len(sys.argv) <= 1:
        print("Supply at least one path to pickled trajectories.")
    else:
        for path in sys.argv[1:]:
            update_traj_file_in_place(path)


if __name__ == "__main__":
    main()
