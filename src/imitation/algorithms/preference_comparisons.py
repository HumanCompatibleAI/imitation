"""Learning reward models using preference comparisons.

Trains a reward model and optionally a policy based on preferences
between trajectory fragments.
"""


import abc
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from imitation.algorithms import base
from imitation.data import rollout
from imitation.data.types import (
    TrajectoryPair,
    TrajectoryWithRew,
    TrajectoryWithRewPair,
    Transitions,
)
from imitation.policies import trainer
from imitation.rewards import reward_nets
from imitation.util import logger


class Fragmenter(abc.ABC):
    """Class for creating pairs of trajectory fragments from a set of trajectories."""

    def __init__(self, custom_logger: Optional[logger.HierarchicalLogger] = None):
        """Initialize the fragmenter.

        Args:
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self.logger = custom_logger or logger.configure()

    @abc.abstractmethod
    def __call__(
        self, trajectories: Sequence[TrajectoryWithRew]
    ) -> Sequence[TrajectoryWithRewPair]:
        """Create fragment pairs out of a sequence of trajectories.

        Args:
            trajectories: collection of trajectories that will be split up into
                fragments

        Returns:
            a sequence of fragment pairs
        """


class RandomFragmenter(Fragmenter):
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
        custom_logger: Optional[logger.HierarchicalLogger] = None,
    ):
        """Initialize the fragmenter.

        Args:
            fragment_length: the length of each sampled fragment
            num_pairs: the number of fragment pairs to sample
            seed: an optional seed for the internal RNG
            warning_threshold: give a warning if the number of available
                transitions is less than this many times the number of
                required samples. Set to 0 to disable this warning.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        super().__init__(custom_logger)
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
        self.logger.log(
            f"Discarded {prev_num_trajectories - len(trajectories)} "
            f"out of {prev_num_trajectories} trajectories because they are "
            f"shorter than the desired length of {self.fragment_length}."
        )

        weights = [len(traj) for traj in trajectories]

        # number of transitions that will be contained in the fragments
        num_transitions = 2 * self.num_pairs * self.fragment_length
        if sum(weights) < num_transitions:
            self.logger.warn(
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
            self.logger.warn(
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
            terminal = (end == n) and traj.terminal
            fragment = TrajectoryWithRew(
                obs=traj.obs[start : end + 1],
                acts=traj.acts[start:end],
                infos=traj.infos[start:end] if traj.infos is not None else None,
                rews=traj.rews[start:end],
                terminal=terminal,
            )
            fragments.append(fragment)
        # fragments is currently a list of single fragments. We want to pair up
        # fragments to get a list of (fragment1, fragment2) tuples. To do so,
        # we create a single iterator of the list and zip it with itself:
        iterator = iter(fragments)
        return list(zip(iterator, iterator))


class PreferenceGatherer(abc.ABC):
    def __init__(self, custom_logger: Optional[logger.HierarchicalLogger] = None):
        """Initialize the preference gatherer.

        Args:
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self.logger = custom_logger or logger.configure()

    @abc.abstractmethod
    def __call__(self, fragment_pairs: Sequence[TrajectoryWithRewPair]) -> np.ndarray:
        """Gathers the probabilities that fragment 1 is preferred
        for a batch of fragments.

        Args:
            fragment_pairs: sequence of pairs of trajectory fragments

        Returns:
            A numpy array with shape (b, ), where b is the length of the input
            (i.e. batch size). Each item in the array is the probability that
            fragment 1 is preferred over fragment 2 for the corresponding
            pair of fragments.

            Note that for human feedback, these probabilities are simply 0 or 1
            (or 0.5 in case of indifference), but synthetic models may yield other
            probabilities.
        """


class SyntheticGatherer(PreferenceGatherer):
    """Computes synthetic preferences using ground-truth environment rewards."""

    def __init__(
        self,
        temperature: float = 1,
        discount_factor: float = 1,
        seed: int = 0,
    ):
        """Initialize the synthetic preference gatherer.

        Args:
            temperature: the preferences are sampled from a softmax, this is
                the temperature used for sampling. temperature=0 leads to deterministic
                results (for equal rewards, 0.5 will be returned).
            discount_factor: discount factor that is used to compute
                how good a fragment is. Default is to use undiscounted
                sums of rewards (as in the DRLHP paper).
            seed: seed for the internal RNG (only used if temperature > 0)
        """
        # we don't pass a logger for now since this particular implementation
        # doesn't use one at the moment
        super().__init__()
        self.temperature = temperature
        self.discount_factor = discount_factor
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, fragment_pairs: Sequence[TrajectoryWithRewPair]) -> np.ndarray:
        rews1, rews2 = self._reward_sums(fragment_pairs)
        if self.temperature == 0:
            return (np.sign(rews1 - rews2) + 1) / 2

        rews1 /= self.temperature
        rews2 /= self.temperature
        # Instead of computing exp(rews1) / (exp(rews1) + exp(rews2)) directly,
        # we divide enumerator and denominator by exp(rews1) to prevent overflows:
        model_probs = 1 / (1 + np.exp(rews2 - rews1))
        return self.rng.binomial(n=1, p=model_probs).astype(np.float32)

    def _reward_sums(self, fragment_pairs) -> Tuple[np.ndarray, np.ndarray]:
        rews1, rews2 = zip(
            *[
                (
                    rollout.compute_returns(f1.rews, self.discount_factor),
                    rollout.compute_returns(f2.rews, self.discount_factor),
                )
                for f1, f2 in fragment_pairs
            ]
        )
        return np.array(rews1), np.array(rews2)


class PreferenceDataset(th.utils.data.Dataset):
    """A PyTorch Dataset for preference comparisons.

    Each item is a tuple consisting of two trajectory fragments
    and a probability that fragment 1 is preferred over fragment 2.

    This dataset is meant to be generated piece by piece during the
    training process, which is why data can be added via the .push()
    method.

    TODO(ejnnr): it should also be possible to store a dataset on disk
    and load it again.
    """

    def __init__(self):
        self.fragments1: List[TrajectoryWithRew] = []
        self.fragments2: List[TrajectoryWithRew] = []
        self.preferences = np.array([])

    def push(
        self,
        fragments: Sequence[TrajectoryWithRewPair],
        preferences: np.ndarray,
    ):
        """Add more samples to the dataset.

        Args:
            fragments: list of pairs of trajectory fragments to add
            preferences: corresponding preference probabilities (probability
                that fragment 1 is preferred over fragment 2)
        """
        fragments1, fragments2 = zip(*fragments)
        if preferences.shape != (len(fragments),):
            raise ValueError(
                f"Unexpected preferences shape {preferences.shape}, "
                f"expected {(len(fragments), )}"
            )
        if preferences.dtype != np.float32:
            raise ValueError("preferences should have dtype float32")

        self.fragments1.extend(fragments1)
        self.fragments2.extend(fragments2)
        self.preferences = np.concatenate((self.preferences, preferences))

    def __getitem__(self, i) -> Tuple[TrajectoryWithRewPair, float]:
        return (self.fragments1[i], self.fragments2[i]), self.preferences[i]

    def __len__(self) -> int:
        assert len(self.fragments1) == len(self.fragments2) == len(self.preferences)
        return len(self.fragments1)


def preference_collate_fn(
    batch: Sequence[Tuple[TrajectoryWithRewPair, float]]
) -> Tuple[Sequence[TrajectoryWithRewPair], Sequence[float]]:
    fragment_pairs, preferences = zip(*batch)
    return list(fragment_pairs), list(preferences)


class RewardTrainer(abc.ABC):
    """Abstract base class for training reward models using preference comparisons.

    This class contains only the actual reward model training code,
    it is not responsible for gathering trajectories and preferences
    or for agent training (see PreferenceComparisons for that).
    """

    def __init__(
        self,
        model: reward_nets.RewardNet,
        custom_logger: Optional[logger.HierarchicalLogger] = None,
    ):
        """Initialize the reward trainer.

        Args:
            model: the RewardNet instance to be trained
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self.model = model
        self.logger = custom_logger or logger.configure()

    @abc.abstractmethod
    def train(self, dataset: PreferenceDataset):
        """Train the reward model on a batch of fragment pairs and preferences.

        Args:
            dataset: the dataset of preference comparisons to train on.
        """


class CrossEntropyRewardTrainer(RewardTrainer):
    """Train a reward model using a cross entropy loss."""

    def __init__(
        self,
        model: reward_nets.RewardNet,
        noise_prob: float = 0.0,
        batch_size: int = 32,
        discount_factor: float = 1.0,
        custom_logger: Optional[logger.HierarchicalLogger] = None,
    ):
        """Initialize the reward model trainer.

        Args:
            model: the RewardNet instance to be trained
            noise_prob: assumed probability with which the preference
                is uniformly random (used for the model of preference generation
                that is used for the loss)
            batch_size: number of fragment pairs per batch
            discount_factor: the model of preference generation uses a softmax
                of returns as the probability that a fragment is preferred.
                This is the discount factor used to calculate those returns.
                Default is 1, i.e. undiscounted sums of rewards (which is what
                the DRLHP paper uses).
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        super().__init__(model, custom_logger)
        self.noise_prob = noise_prob
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.optim = th.optim.Adam(self.model.parameters())

    def _loss(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
    ) -> th.Tensor:
        probs = th.empty(len(fragment_pairs), dtype=th.float32)
        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])
            rews1 = self._rewards(trans1)
            rews2 = self._rewards(trans2)
            probs[i] = self._probability(rews1, rews2)
        return th.nn.functional.binary_cross_entropy(
            probs, th.as_tensor(preferences, dtype=th.float32)
        )

    def _rewards(self, transitions: Transitions) -> th.Tensor:
        return self.model(*self.model.preprocess(transitions))

    def _probability(self, rews1: th.Tensor, rews2: th.Tensor) -> th.Tensor:
        assert rews1.ndim == rews2.ndim == 1
        # First, we compute the difference of the returns of
        # the two fragments. We have a special case for a discount
        # factor of 1 to avoid unnecessary computation (especially
        # since this is the default setting).
        if self.discount_factor == 1:
            returns_diff = (rews2 - rews1).sum()
        else:
            discounts = self.discount_factor ** th.arange(len(rews1))
            returns_diff = (discounts * (rews2 - rews1)).sum()
        # We take the softmax of the returns. model_probability
        # is the first dimension of that softmax, representing the
        # probability that fragment 1 is preferred.
        model_probability = 1 / (1 + returns_diff.exp())
        return self.noise_prob * 0.5 + (1 - self.noise_prob) * model_probability

    def train(self, dataset: PreferenceDataset):
        # TODO(ejnnr): This isn't specific to the loss function or probability model.
        # In general, it might be best to split the probability model, the loss and
        # the optimization procedure a bit more cleanly so that different versions
        # can be combined

        dataloader = th.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=th.utils.data.RandomSampler(
                dataset, replacement=True, num_samples=None
            ),
            collate_fn=preference_collate_fn,
        )
        for fragment_pairs, preferences in dataloader:
            self.optim.zero_grad()
            loss = self._loss(fragment_pairs, preferences)
            loss.backward()
            self.optim.step()
            self.logger.record("reward/loss", loss.item())


class PreferenceComparisons(base.BaseImitationAlgorithm):
    """Main interface for reward learning using preference comparisons."""

    def __init__(
        self,
        trajectory_generator: trainer.TrajectoryGenerator,
        reward_model: reward_nets.RewardNet,
        sample_steps: int,
        agent_steps: Optional[int] = None,
        fragmenter: Optional[Fragmenter] = None,
        preference_gatherer: Optional[PreferenceGatherer] = None,
        reward_trainer: Optional[RewardTrainer] = None,
        custom_logger: Optional[logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False,
    ):
        """Initialize the preference comparison trainer.

        The loggers of all subcomponents are overridden with the logger used
        by this class.

        Args:
            trajectory_generator: generates trajectories while optionally training
                an RL agent on the learned reward function (can also be a sampler
                from a static dataset of trajectories though).
            reward_model: a RewardNet instance to be used for learning the reward
            sample_steps: number of environment timesteps to sample
                for creating fragments.
            agent_steps: number of environment steps to train the agent for between
                each reward model training round. Defaults to sample_steps.
            fragmenter: takes in a set of trajectories and returns pairs of fragments
                for which preferences will be gathered. These fragments could be random,
                or they could be selected more deliberately (active learning).
                Default is a random fragmenter.
            preference_gatherer: how to get preferences between trajectory fragments.
                Default (and currently the only option) is to use synthetic preferences
                based on ground-truth rewards. Human preferences could be implemented
                here in the future.
            reward_trainer: trains the reward model based on pairs of fragments and
                associated preferences. Default is to use the preference model
                and loss function from DRLHP.
            custom_logger: Where to log to; if None (default), creates a new logger.
            allow_variable_horizon: If False (default), algorithm will raise an
                exception if it detects trajectories of different length during
                training. If True, overrides this safety check. WARNING: variable
                horizon episodes leak information about the reward via termination
                condition, and can seriously confound evaluation. Read
                https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
                before overriding this.
        """
        super().__init__(
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )

        self.model = reward_model
        self.reward_trainer = reward_trainer or CrossEntropyRewardTrainer(
            reward_model, custom_logger=self.logger
        )
        # If the reward trainer was created in the previous line, we've already passed
        # the correct logger. But if the user created a RewardTrainer themselves and
        # didn't manually set a logger, it would be annoying if a separate one was used.
        self.reward_trainer.logger = self.logger
        # the reward_trainer's model should refer to the same object as our copy
        assert self.reward_trainer.model is self.model
        self.trajectory_generator = trajectory_generator
        self.trajectory_generator.logger = self.logger
        self.fragmenter = fragmenter or RandomFragmenter(custom_logger=self.logger)
        self.fragmenter.logger = self.logger
        self.preference_gatherer = preference_gatherer or SyntheticGatherer()
        self.preference_gatherer.logger = self.logger
        self.sample_steps = sample_steps
        # In contrast to the previous cases, we need the is None check
        # because someone might explicitly set agent_timesteps=0.
        if agent_steps is None:
            agent_steps = sample_steps
        self.agent_steps = agent_steps
        self.dataset = PreferenceDataset()

    def train(self, iterations: int):
        """Train the reward model and the policy if applicable.

        Args:
            iterations: number of iterations of the outer training loop
        """
        for _ in range(iterations):
            self.logger.log(f"Collecting {self.sample_steps} trajectory steps")
            trajectories = self.trajectory_generator.sample(self.sample_steps)
            self._check_fixed_horizon(trajectories)
            self.logger.log("Creating fragment pairs")
            fragments = self.fragmenter(trajectories)
            self.logger.log("Gathering preferences")
            preferences = self.preference_gatherer(fragments)
            self.dataset.push(fragments, preferences)
            self.logger.log(f"Dataset now contains {len(self.dataset)} samples")
            self.logger.log("Training reward model")
            self.reward_trainer.train(self.dataset)
            self.logger.log(f"Training agent for {self.agent_steps} steps")
            self.trajectory_generator.train(steps=self.agent_steps)
