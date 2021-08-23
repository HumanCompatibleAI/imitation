import random
import warnings
from typing import Callable, List, Optional, Tuple

from imitation.data.types import TrajectoryWithRew

Fragmenter = Callable[
    [List[TrajectoryWithRew]], List[Tuple[TrajectoryWithRew, TrajectoryWithRew]]
]
"""Creates pairs of trajectory fragments from a collection of trajectories."""


class RandomFragmenter:
    """Sample fragments of trajectories uniformly at random with replacement.

    Note that each fragment is part of a single episode and has a fixed
    length. This leads to a bias: transitions at the beginning and at the
    end of episodes are less likely to occur as part of fragments (this affects
    the first and last fragment_length transitions).

    TODO(ejnnr): should we correct for this bias and make all transitions equally
    likely to be part of fragments?
    """

    def __init__(
        self,
        fragment_length: int = 50,
        num_pairs: int = 50,
        seed: Optional[float] = None,
    ):
        """Initialize the fragmenter.

        Args:
            fragment_length: the length of each sampled fragment
            num_pairs: the number of fragment pairs to sample
            seed: an optional seed for the internal RNG
        """
        self.fragment_length = fragment_length
        self.num_pairs = num_pairs
        self.rng = random.Random(seed)

    def __call__(
        self, trajectories: List[TrajectoryWithRew]
    ) -> List[Tuple[TrajectoryWithRew, TrajectoryWithRew]]:
        fragments: List[TrajectoryWithRew] = []

        # filter out all trajectories that are too short
        trajectories = [
            traj for traj in trajectories if len(traj) >= self.fragment_length
        ]
        if len(trajectories) == 0:
            raise ValueError(
                "No trajectories are long enough for the desired fragment length."
            )

        weights = [len(traj) for traj in trajectories]

        # TODO(ejnnr): since we're sampling with replacement, there could
        # already be lots of duplicates below this threshold. Perhaps
        # we should warn already if the number of transitions isn't larger
        # than what's needed by a factor of e.g. 10?
        # Alternatively, sample without replacement, but that seems much
        # trickier to do (and maybe introduces new biases?)
        if sum(weights) < 2 * self.num_pairs * self.fragment_length:
            warnings.warn(
                "Fewer transitions available than needed for desired number "
                "of fragment pairs. Some transitions will appear multiple times."
            )

        # we need two fragments for each comparison
        for _ in range(2 * self.num_pairs):
            traj = self.rng.choices(trajectories, weights, k=1)[0]
            n = len(traj)
            start = self.rng.randint(0, n - self.fragment_length)
            end = start + self.fragment_length
            fragment = TrajectoryWithRew(
                obs=traj.obs[start : end + 1],
                acts=traj.acts[start:end],
                infos=traj.infos[start:end] if traj.infos is not None else None,
                rews=traj.rews[start:end],
            )
            fragments.append(fragment)
        # fragments is currently a list of single fragments. We want to pair up
        # fragments to get a list of (fragment1, fragment2) tuples. To do so,
        # we create a single iterator of the list and zip it with itself:
        iterator = iter(fragments)
        return list(zip(iterator, iterator))
