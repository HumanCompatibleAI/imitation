"""Learning reward models using preference comparisons.

Trains a reward model and optionally a policy based on preferences
between trajectory fragments.
"""


import abc
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch as th
from stable_baselines3.common import logger

from imitation.data import fragments, rollout
from imitation.data.types import (
    TrajectoryPair,
    TrajectoryWithRew,
    TrajectoryWithRewPair,
    Transitions,
)
from imitation.policies import trainer
from imitation.rewards import reward_nets

PreferenceGatherer = Callable[[Sequence[TrajectoryWithRewPair]], np.ndarray]
"""Gathers the probabilities that fragment 1 is preferred for a batch of fragments.

Takes a list of pairs of trajectory fragments as input. Should return a numpy
array with shape (b, ), where b is the length of the list (i.e. batch size).
Each item in the array is the probability that fragment 1 is preferred over
fragment 2 for the corresponding pair of fragments.

Note that for human feedback, these probabilities are simply 0 or 1
(or 0.5 in case of indifference), but synthetic models may yield other
probabilities.
"""


class SyntheticGatherer:
    """Computes synthetic preferences using ground-truth environment rewards."""

    def __init__(
        self,
        probabilistic: bool = True,
        noise_prob: float = 0.0,
        discount_factor: float = 1,
        seed: int = 0,
    ):
        """Initialize the synthetic preference gatherer.

        Args:
            probabilistic: if True (default), the outputs are sampled from a Bernoulli
                distribution based on a softmax of the rewards (e.g. if the fragments
                have equal rewards, 0 or 1 is returned with equal probability).
                If False, then 0, 1 or 0.5 is returned deterministically based on which
                fragment is better or whether they are equally good.
            noise_prob: probability with which the preference is uniformly random
                rather than based on the softmax of rewards.
                Cannot be set if probabilistic=False.
            discount_factor: discount factor that is used to compute
                how good a fragment is. Default is to use undiscounted
                sums of rewards (as in the DRLHP paper).
            seed: seed for the internal RNG (only used if probabilistic is True).
        """
        self.noise_prob = noise_prob
        self.probabilistic = probabilistic
        self.discount_factor = discount_factor
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, fragment_pairs: Sequence[TrajectoryWithRewPair]) -> np.ndarray:
        rews1, rews2 = self._reward_sums(fragment_pairs)
        if self.probabilistic:
            # If probabilistic is True, we use the human model to compute probabilities.
            # Instead of computing exp(rews1) / (exp(rews1) + exp(rews2)) directly,
            # we divide enumerator and denominator by exp(rews1) to prevent overflows:
            model_probs = 1 / (1 + np.exp(rews2 - rews1))
            # We also include an optional probability of making a random decision
            # (modeling the fact that humans make mistakes):
            noisy_probs = self.noise_prob * 0.5 + (1 - self.noise_prob * model_probs)
            return self.rng.binomial(n=1, p=noisy_probs).astype(np.float32)

        if self.noise_prob != 0:
            raise ValueError(
                f"noise_prob={self.noise_prob} is non-zero but probabilistic=False"
            )
        # if probabilistic is set to False, we simply return 0 or 1 (or 0.5 if the sums
        # of rewards are the same) for the probability
        return (np.sign(rews1 - rews2) + 1) / 2

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

    def __init__(self, model: reward_nets.RewardNet):
        """Initialize the reward trainer.

        Args:
            model: the RewardNet instance to be trained
        """
        self.model = model

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
        """
        super().__init__(model)
        self.noise_prob = noise_prob
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.optim = th.optim.Adam(self.model.parameters())

    def _loss(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
    ):
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
        # In general, it mighe be best to split the probability model, the loss and
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
            logger.record("reward/loss", loss.item())


class PreferenceComparisons:
    """Main interface for reward learning using preference comparisons."""

    def __init__(
        self,
        trajectory_generator: trainer.TrajectoryGenerator,
        reward_model: reward_nets.RewardNet,
        timesteps: int,
        agent_timesteps: Optional[int] = None,
        fragmenter: Optional[fragments.Fragmenter] = None,
        preference_gatherer: Optional[PreferenceGatherer] = None,
        reward_trainer: Optional[RewardTrainer] = None,
    ):
        """Initialize the preference comparison trainer.

        Args:
            trajectory_generator: generates trajectories while optionally training
                an RL agent on the learned reward function (can also be a sampler
                from a static dataset of trajectories though).
            reward_model: a RewardNet instance to be used for learning the reward
            timesteps: number of environment timesteps to sample for creating fragments.
            agent_timesteps: number of environment steps to train the agent for between
                each reward model training round. Defaults to timesteps.
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
        """
        self.model = reward_model
        self.reward_trainer = reward_trainer or CrossEntropyRewardTrainer(reward_model)
        # the reward_trainer's model should refer to the same object as our copy
        assert self.reward_trainer.model is self.model
        self.trajectory_generator = trajectory_generator
        self.fragmenter = fragmenter or fragments.RandomFragmenter()
        self.preference_gatherer = preference_gatherer or SyntheticGatherer()
        self.timesteps = timesteps
        if agent_timesteps is None:
            agent_timesteps = timesteps
        # In contrast to the previous cases, we need the is None check
        # because someone might explicitly set agent_timesteps=0.
        self.agent_timesteps = agent_timesteps
        self.dataset = PreferenceDataset()

    def train(self, steps: int):
        """Train the reward model and the policy if applicable.

        Args:
            steps: number of iterations of the outer training loop
        """
        for _ in range(steps):
            logger.log(f"Training agent for {self.agent_timesteps} steps")
            self.trajectory_generator.train(steps=self.agent_timesteps)
            logger.log(f"Collecting {self.timesteps} trajectory steps")
            trajectories = self.trajectory_generator.sample(self.timesteps)
            logger.log(f"Creating {self.fragmenter.num_pairs} fragment pairs")
            fragments = self.fragmenter(trajectories)
            logger.log("Gathering preferences")
            preferences = self.preference_gatherer(fragments)
            self.dataset.push(fragments, preferences)
            logger.log(f"Dataset now contains {len(self.dataset)} samples")
            logger.log("Training reward model")
            self.reward_trainer.train(self.dataset)
