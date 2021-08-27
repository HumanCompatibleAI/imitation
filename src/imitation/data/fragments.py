import random
import warnings
from typing import Callable, List, Optional, Sequence

from stable_baselines3.common import logger

from imitation.data.types import TrajectoryWithRew, TrajectoryWithRewPair

Fragmenter = Callable[[Sequence[TrajectoryWithRew]], Sequence[TrajectoryWithRewPair]]
"""Creates pairs of trajectory fragments from a collection of trajectories."""


class RandomFragmenter:
    """Sample fragments of trajectories uniformly at random with replacement.

    Note that each fragment is part of a single episode and has a fixed
    length. This leads to a bias: transitions at the beginning and at the
    end of episodes are less likely to occur as part of fragments (this affects
    the first and last fragment_length transitions).

    An additional bias is that trajectories shorter than the desired fragment
    length are never used.
    """

    def __init__(
        self,
        fragment_length: int = 50,
        num_pairs: int = 50,
        seed: Optional[float] = None,
        warning_threshold: int = 10,
    ):
        """Initialize the fragmenter.

        Args:
            fragment_length: the length of each sampled fragment
            num_pairs: the number of fragment pairs to sample
            seed: an optional seed for the internal RNG
            warning_threshold: give a warning if the number of available
                transitions is less than this many times the number of
                required samples. Set to 0 to disable this warning.
        """
        self.fragment_length = fragment_length
        self.num_pairs = num_pairs
        self.rng = random.Random(seed)
        self.warning_threshold = warning_threshold

    def __call__(
        self, trajectories: Sequence[TrajectoryWithRew]
    ) -> Sequence[TrajectoryWithRewPair]:
        fragments: List[TrajectoryWithRew] = []

        prev_num_trajectories = len(trajectories)
        # filter out all trajectories that are too short
        trajectories = [
            traj for traj in trajectories if len(traj) >= self.fragment_length
        ]
        if len(trajectories) == 0:
            raise ValueError(
                "No trajectories are long enough for the desired fragment length."
            )
        logger.log(
            f"Discarded {prev_num_trajectories - len(trajectories)} "
            f"out of {prev_num_trajectories} trajectories because they are "
            f"shorter than the desired length of {self.fragment_length}."
        )

        weights = [len(traj) for traj in trajectories]

        # number of transitions that will be contained in the fragments
        num_transitions = 2 * self.num_pairs * self.fragment_length
        if sum(weights) < num_transitions:
            warnings.warn(
                "Fewer transitions available than needed for desired number "
                "of fragment pairs. Some transitions will appear multiple times."
            )
        elif (
            self.warning_threshold
            and sum(weights) < self.warning_threshold * num_transitions
        ):
            # If the number of available transitions is not much larger
            # than the number of requires ones, we already give a warning.
            # But only if self.warning_threshold is non-zero.
            warnings.warn(
                f"Samples will contain {num_transitions} transitions in total "
                f"and only {sum(weights)} are available. "
                f"Because we sample with replacement, a significant number "
                "of transitions are likely to appear multiple times."
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
