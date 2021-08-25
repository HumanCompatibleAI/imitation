"""Learning reward models using preference comparisons.

Trains a reward model and optionally a policy based on preferences
between trajectory fragments.
"""


import abc
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch as th

from imitation.data.fragments import Fragmenter, RandomFragmenter
from imitation.data.rollout import flatten_trajectories_with_rew
from imitation.data.types import TrajectoryWithRew, TransitionsWithRew
from imitation.policies.trainer import AgentTrainer
from imitation.rewards.reward_nets import RewardNet

PreferenceGatherer = Callable[
    [List[Tuple[TrajectoryWithRew, TrajectoryWithRew]]], np.ndarray
]
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

    def __init__(self, probabilistic: bool = False, noise_prob: float = 0.0):
        """Initialize the synthetic preference gatherer.

        Args:
            probabilistic: if False (default), the preferences are either 1
                (if fragment 1 has a higher sum of rewards), 0 (if fragment 2
                has a higher sum of rewards) or 0.5 (if they are equal).
                If True, a probability between 0 and 1 is returned
                instead (based on a softmax).
            noise_prob: probability with which the preference is uniformly random.
                This is incorporated into the model if probabilistic=True,
                no random decisions are actually made! Ignored if probabilistic=False.
        """
        self.noise_prob = noise_prob
        self.probabilistic = probabilistic

    def __call__(
        self, fragments: List[Tuple[TrajectoryWithRew, TrajectoryWithRew]]
    ) -> np.ndarray:
        rews1, rews2 = self._reward_sums(fragments)
        if self.probabilistic:
            # If probabilistic is True, we use the human model to compute probabilities.
            # Instead of computing exp(rews1) / (exp(rews1) + exp(rews2)) directly,
            # we divide enumerator and denominator by exp(rews1) to prevent overflows:
            model_probabilities = 1 / (1 + np.exp(rews2 - rews1))
            # We also include an optional probability of making a random decision
            # (modeling the fact that humans make mistakes):
            return self.noise_prob * 0.5 + (1 - self.noise_prob * model_probabilities)
        if self.noise_prob != 0:
            raise ValueError(
                f"noise_prob={self.noise_prob} is non-zero but probabilistic=False"
            )
        # if probabilistic is set to False, we simply return 0 or 1 (or 0.5 if the sums
        # of rewards are the same) for the probability
        return (np.sign(rews1 - rews2) + 1) / 2

    def _reward_sums(self, fragments) -> Tuple[np.ndarray, np.ndarray]:
        rews1, rews2 = zip(
            *[(np.sum(f1.rews), np.sum(f2.rews)) for f1, f2 in fragments]
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
        fragments: List[Tuple[TrajectoryWithRew, TrajectoryWithRew]],
        preferences: np.ndarray,
    ):
        """Add more samples to the dataset.

        Args:
            fragments: list of pairs of trajectory fragments to add
            preferences: corresponding preference probabilities (probabilities
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

    def __getitem__(self, i) -> Tuple[TrajectoryWithRew, TrajectoryWithRew, float]:
        return self.fragments1[i], self.fragments2[i], self.preferences[i]

    def __len__(self) -> int:
        return len(self.fragments1)


def preference_collate_fn(
    batch: List[Tuple[TrajectoryWithRew, TrajectoryWithRew, float]]
):
    fragments1, fragments2, preferences = zip(*batch)
    return list(zip(fragments1, fragments2)), list(preferences)


class RewardTrainer(abc.ABC):
    """Abstract base class for training reward models using preference comparisons.

    This class contains only the actual reward model training code,
    it is not responsible for gathering trajectories and preferences
    or for agent training (see PreferenceComparisons for that).
    """

    def __init__(self, model: RewardNet):
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
    """Train a reward model using a cross entropy loss.

    Args:
        model: the RewardNet instance to be trained
        noise_prob: assumed probability with which the preference is uniformly random
            (used for the model of preference generation that is used for the loss)
    """

    def __init__(
        self,
        model: RewardNet,
        noise_prob: float = 0.0,
        batch_size: int = 32,
    ):
        super().__init__(model)
        self.noise_prob = noise_prob
        self.batch_size = batch_size
        self.optim = th.optim.Adam(self.model.parameters())

    def _loss(
        self,
        fragments: List[Tuple[TrajectoryWithRew, TrajectoryWithRew]],
        preferences: np.ndarray,
    ):
        probs = th.empty(len(fragments), dtype=th.float32)
        for i, fragment in enumerate(fragments):
            frag1, frag2 = fragment
            trans1 = flatten_trajectories_with_rew([frag1])
            trans2 = flatten_trajectories_with_rew([frag2])
            rews1 = self._rewards(trans1)
            rews2 = self._rewards(trans2)
            probs[i] = self._probability(rews1, rews2)
        return th.nn.functional.binary_cross_entropy(
            probs, th.as_tensor(preferences, dtype=th.float32)
        )

    def _rewards(self, transitions: TransitionsWithRew) -> th.Tensor:
        return self.model(*self.model.preprocess(transitions))

    def _probability(self, rews1: th.Tensor, rews2: th.Tensor) -> th.Tensor:
        model_probability = 1 / (1 + (rews2 - rews1).sum().exp())
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
        self.optim.zero_grad()
        for fragments, preferences in dataloader:
            loss = self._loss(fragments, preferences)
            loss.backward()
        self.optim.step()


class PreferenceComparisons:
    """Main interface for reward learning using preference comparisons."""

    def __init__(
        self,
        agent_trainer: AgentTrainer,
        reward_model: RewardNet,
        agent_timesteps: int,
        fragmenter: Optional[Fragmenter] = None,
        preference_gatherer: Optional[PreferenceGatherer] = None,
        reward_trainer: Optional[RewardTrainer] = None,
    ):
        """Initialize the preference comparison trainer.

        Args:
            agent_trainer: generates trajectories while optionally training an RL agent
                on the learned reward function (can also be a sampler from a static
                dataset of trajectories though).
            reward_model: a RewardNet instance to be used for learning the reward
            agent_timesteps: number of environment steps to train the agent for between
                each reward model training round
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
        if reward_trainer is None:
            reward_trainer = CrossEntropyRewardTrainer(reward_model)
        self.reward_trainer = reward_trainer
        self.agent = agent_trainer
        if fragmenter is None:
            fragmenter = RandomFragmenter()
        self.fragmenter = fragmenter
        if preference_gatherer is None:
            preference_gatherer = SyntheticGatherer()
        self.preference_gatherer = preference_gatherer
        self.agent_timesteps = agent_timesteps

        self.dataset = PreferenceDataset()

    def train(self, steps: int):
        """Train the reward model and the policy if applicable.

        Args:
            steps: number of iterations of the outer training loop
        """
        for _ in range(steps):
            trajectories = self.agent.train(total_timesteps=self.agent_timesteps)
            fragments = self.fragmenter(trajectories)
            preferences = self.preference_gatherer(fragments)
            self.dataset.push(fragments, preferences)
            self.reward_trainer.train(self.dataset)
