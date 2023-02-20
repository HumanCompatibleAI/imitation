"""Converts old-style pickle or npz trajectories to new-style HuggingFace datasets.

See https://github.com/HumanCompatibleAI/imitation/pull/448 for a description
of the new trajectory format.

This script takes as command-line input multiple paths to saved trajectories,
in the old .pkl or .npz format. It then saves each sequence in the new HuggingFace
datasets format. The path is the same as the original but a directory without an
extension (i.e. "A.pkl" -> "A/", "A.npz" -> "A/", "A/" -> "A/", "A.foo" -> "A/").
"""
import pathlib
import warnings

import imitation.data.serialize
from imitation.data import types


def update_traj_file_in_place(path_str: types.AnyPath, /) -> pathlib.Path:
    """Converts pickle or npz file to the new HuggingFace format.

    The new data is saved as `Sequence[imitation.types.TrajectoryWithRew]`.

    Args:
        path_str: Path to a pickle or npz file containing
            `Sequence[imitation.types.Trajectory]`
            or `Sequence[imitation.old_types.TrajectoryWithRew]`.

    Returns:
        The path to the converted trajectory dataset.
    """
    path = imitation.data.serialize.parse_path(path_str)
    with warnings.catch_warnings():
        # Filter out DeprecationWarning because we expect to load old trajectories here.
        warnings.filterwarnings(
            "ignore",
            message="Loading old .* version of Trajectories.*",
            category=DeprecationWarning,
        )
        trajs = imitation.data.serialize.load(path)

    if isinstance(
        trajs,
        imitation.data.huggingface_datasets_conversion.TrajectoryDatasetSequence,
    ):
        warnings.warn(f"File {path} is already in the new format. Skipping.")
        return path
    else:
        converted_path = path.with_suffix("")
        imitation.data.serialize.save(converted_path, trajs)
        return converted_path


def main():
    import sys

    if len(sys.argv) <= 1:
        print("Supply at least one path to pickled trajectories.")
    else:
        for path in sys.argv[1:]:
            update_traj_file_in_place(path)


if __name__ == "__main__":
    main()
