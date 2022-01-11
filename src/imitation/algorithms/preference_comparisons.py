"""Learning reward models using preference comparisons.

Trains a reward model and optionally a policy based on preferences
between trajectory fragments.
"""
import abc
import math
import pickle
import random
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th
from scipy import special
from stable_baselines3.common import base_class, vec_env

from imitation.algorithms import base
from imitation.data import rollout, types, wrappers
from imitation.data.types import (
    AnyPath,
    TrajectoryPair,
    TrajectoryWithRew,
    TrajectoryWithRewPair,
    Transitions,
)
from imitation.policies import exploration_wrapper
from imitation.rewards import common as rewards_common
from imitation.rewards import reward_nets, reward_wrapper
from imitation.util import logger as imit_logger


class TrajectoryGenerator(abc.ABC):
    """Generator of trajectories with optional training logic."""

    _logger: imit_logger.HierarchicalLogger
    """Object to log statistics and natural language messages to."""

    def __init__(self, custom_logger: Optional[imit_logger.HierarchicalLogger] = None):
        """Builds TrajectoryGenerator.

        Args:
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self._logger = custom_logger or imit_logger.configure()

    @abc.abstractmethod
    def sample(self, steps: int) -> Sequence[TrajectoryWithRew]:
        """Sample a batch of trajectories.

        Args:
            steps: All trajectories taken together should
                have at least this many steps.

        Returns:
            A list of sampled trajectories with rewards (which should
            be the environment rewards, not ones from a reward model).
        """  # noqa: DAR202

    def train(self, steps: int, **kwargs):
        """Train an agent if the trajectory generator uses one.

        By default, this method does nothing and doesn't need
        to be overridden in subclasses that don't require training.

        Args:
            steps: number of environment steps to train for.
            **kwargs: additional keyword arguments to pass on to
                the training procedure.
        """

    @property
    def logger(self) -> imit_logger.HierarchicalLogger:
        return self._logger

    @logger.setter
    def logger(self, value: imit_logger.HierarchicalLogger):
        self._logger = value


class TrajectoryDataset(TrajectoryGenerator):
    """A fixed dataset of trajectories."""

    def __init__(
        self,
        path: AnyPath,
        seed: Optional[int] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Creates a dataset loaded from `path`.

        Args:
            path: A path to pickled rollouts.
            seed: Seed for RNG used for shuffling dataset.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        super().__init__(custom_logger=custom_logger)
        self._trajectories = types.load(path)
        self.rng = random.Random(seed)

    def sample(self, steps: int) -> Sequence[TrajectoryWithRew]:
        # make a copy before shuffling
        trajectories = list(self._trajectories)
        self.rng.shuffle(trajectories)
        return _get_trajectories(trajectories, steps)


class AgentTrainer(TrajectoryGenerator):
    """Wrapper for training an SB3 algorithm on an arbitrary reward function."""

    def __init__(
        self,
        algorithm: base_class.BaseAlgorithm,
        reward_fn: Union[rewards_common.RewardFn, reward_nets.RewardNet],
        exploration_frac: float = 0.0,
        stay_prob: float = 0.5,
        random_prob: float = 0.5,
        seed: Optional[int] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Initialize the agent trainer.

        Args:
            algorithm: the stable-baselines algorithm to use for training.
                Its environment must be set.
            reward_fn: either a RewardFn or a RewardNet instance that will supply
                the rewards used for training the agent.
            exploration_frac: fraction of the trajectories that will be generated
                partially randomly rather than only by the agent when sampling.
            stay_prob: the probability of staying with the current policy at each
                step for the exploratory samples.
            random_prob: the probability of picking the random policy when switching
                during exploration.
            seed: random seed for exploratory trajectories.
            custom_logger: Where to log to; if None (default), creates a new logger.

        Raises:
            ValueError: `algorithm` does not have an environment set.
        """
        self.algorithm = algorithm
        # NOTE: this has to come after setting self.algorithm because super().__init__
        # will set self.logger, which also sets the logger for the algorithm
        super().__init__(custom_logger)
        if isinstance(reward_fn, reward_nets.RewardNet):
            reward_fn = reward_fn.predict
        self.reward_fn = reward_fn
        self.exploration_frac = exploration_frac

        venv = self.algorithm.get_env()
        if not isinstance(venv, vec_env.VecEnv):
            raise ValueError("The environment for the agent algorithm must be set.")
        # The BufferingWrapper records all trajectories, so we can return
        # them after training. This should come first (before the wrapper that
        # changes the reward function), so that we return the original environment
        # rewards.
        self.buffering_wrapper = wrappers.BufferingWrapper(venv)
        self.venv = reward_wrapper.RewardVecEnvWrapper(
            self.buffering_wrapper,
            reward_fn,
        )
        self.algorithm.set_env(self.venv)
        policy = rollout._policy_to_callable(
            self.algorithm,
            self.venv,
            deterministic_policy=True,
        )
        self.exploration_wrapper = exploration_wrapper.ExplorationWrapper(
            policy=policy,
            venv=self.venv,
            random_prob=random_prob,
            stay_prob=stay_prob,
            seed=seed,
        )

    def train(self, steps: int, **kwargs) -> None:
        """Train the agent using the reward function specified during instantiation.

        Args:
            steps: number of environment timesteps to train for
            **kwargs: other keyword arguments to pass to BaseAlgorithm.train()

        Raises:
            RuntimeError: Transitions left in `self.buffering_wrapper`; call
                `self.sample` first to clear them.
        """
        n_transitions = self.buffering_wrapper.n_transitions
        if n_transitions:
            raise RuntimeError(
                f"There are {n_transitions} transitions left in the buffer. "
                "Call AgentTrainer.sample() first to clear them.",
            )
        self.algorithm.learn(total_timesteps=steps, reset_num_timesteps=False, **kwargs)

    def sample(self, steps: int) -> Sequence[types.TrajectoryWithRew]:
        agent_trajs, _ = self.buffering_wrapper.pop_finished_trajectories()
        # We typically have more trajectories than are needed.
        # In that case, we use the final trajectories because
        # they are the ones with the most relevant version of
        # the agent.
        # The easiest way to do this will be to first invert the
        # list and then later just take the first trajectories:
        agent_trajs = agent_trajs[::-1]
        avail_steps = sum(len(traj) for traj in agent_trajs)

        exploration_steps = int(self.exploration_frac * steps)
        if self.exploration_frac > 0 and exploration_steps == 0:
            self.logger.warn(
                "No exploration steps included: exploration_frac = "
                f"{self.exploration_frac} > 0 but steps={steps} is too small.",
            )
        agent_steps = steps - exploration_steps

        if avail_steps < agent_steps:
            self.logger.log(
                f"Requested {agent_steps} transitions but only {avail_steps} in buffer."
                f" Sampling {agent_steps - avail_steps} additional transitions.",
            )
            sample_until = rollout.make_sample_until(
                min_timesteps=agent_steps - avail_steps,
                min_episodes=None,
            )
            # Important note: we don't want to use the trajectories returned
            # here because 1) they might miss initial timesteps taken by the RL agent
            # and 2) their rewards are the ones provided by the reward model!
            # Instead, we collect the trajectories using the BufferingWrapper.
            rollout.generate_trajectories(
                self.algorithm,
                self.venv,
                sample_until=sample_until,
            )
            additional_trajs, _ = self.buffering_wrapper.pop_finished_trajectories()

            agent_trajs = list(agent_trajs) + list(additional_trajs)

        agent_trajs = _get_trajectories(agent_trajs, agent_steps)

        exploration_trajs = []
        if exploration_steps > 0:
            self.logger.log(f"Sampling {exploration_steps} exploratory transitions.")
            sample_until = rollout.make_sample_until(
                min_timesteps=exploration_steps,
                min_episodes=None,
            )
            rollout.generate_trajectories(
                policy=self.exploration_wrapper,
                venv=self.venv,
                sample_until=sample_until,
            )
            exploration_trajs, _ = self.buffering_wrapper.pop_finished_trajectories()
            exploration_trajs = _get_trajectories(exploration_trajs, exploration_steps)

        # We call _get_trajectories separately on agent_trajs and exploration_trajs
        # and then just concatenate. This could mean we return slightly too many
        # transitions, but it gets the proportion of exploratory and agent transitions
        # roughly right.
        return list(agent_trajs) + list(exploration_trajs)

    @TrajectoryGenerator.logger.setter
    def logger(self, value: imit_logger.HierarchicalLogger):
        self._logger = value
        self.algorithm.set_logger(self.logger)


def _get_trajectories(
    trajectories: Sequence[TrajectoryWithRew],
    steps: int,
) -> Sequence[TrajectoryWithRew]:
    """Get enough trajectories to have at least `steps` transitions in total."""
    if steps == 0:
        return []

    available_steps = sum(len(traj) for traj in trajectories)
    if available_steps < steps:
        raise RuntimeError(
            f"Asked for {steps} transitions but only {available_steps} available",
        )
    # We need the cumulative sum of trajectory lengths
    # to determine how many trajectories to return:
    steps_cumsum = np.cumsum([len(traj) for traj in trajectories])
    # Now we find the first index that gives us enough
    # total steps:
    idx = (steps_cumsum >= steps).argmax()
    # we need to include the element at position idx
    trajectories = trajectories[: idx + 1]
    # sanity check
    assert sum(len(traj) for traj in trajectories) >= steps
    return trajectories


class Fragmenter(abc.ABC):
    """Class for creating pairs of trajectory fragments from a set of trajectories."""

    def __init__(self, custom_logger: Optional[imit_logger.HierarchicalLogger] = None):
        """Initialize the fragmenter.

        Args:
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self.logger = custom_logger or imit_logger.configure()

    @abc.abstractmethod
    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        """Create fragment pairs out of a sequence of trajectories.

        Args:
            trajectories: collection of trajectories that will be split up into
                fragments
            fragment_length: the length of each sampled fragment
            num_pairs: the number of fragment pairs to sample

        Returns:
            a sequence of fragment pairs
        """  # noqa: DAR202


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
        seed: Optional[float] = None,
        warning_threshold: int = 10,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Initialize the fragmenter.

        Args:
            seed: an optional seed for the internal RNG
            warning_threshold: give a warning if the number of available
                transitions is less than this many times the number of
                required samples. Set to 0 to disable this warning.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        super().__init__(custom_logger)
        self.rng = random.Random(seed)
        self.warning_threshold = warning_threshold

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        fragments: List[TrajectoryWithRew] = []

        prev_num_trajectories = len(trajectories)
        # filter out all trajectories that are too short
        trajectories = [traj for traj in trajectories if len(traj) >= fragment_length]
        if len(trajectories) == 0:
            raise ValueError(
                "No trajectories are long enough for the desired fragment length "
                f"of {fragment_length}.",
            )
        num_discarded = prev_num_trajectories - len(trajectories)
        if num_discarded:
            self.logger.log(
                f"Discarded {num_discarded} out of {prev_num_trajectories} "
                "trajectories because they are shorter than the desired length "
                f"of {fragment_length}.",
            )

        weights = [len(traj) for traj in trajectories]

        # number of transitions that will be contained in the fragments
        num_transitions = 2 * num_pairs * fragment_length
        if sum(weights) < num_transitions:
            self.logger.warn(
                "Fewer transitions available than needed for desired number "
                "of fragment pairs. Some transitions will appear multiple times.",
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
                "of transitions are likely to appear multiple times.",
            )

        # we need two fragments for each comparison
        for _ in range(2 * num_pairs):
            traj = self.rng.choices(trajectories, weights, k=1)[0]
            n = len(traj)
            start = self.rng.randint(0, n - fragment_length)
            end = start + fragment_length
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
    """Base class for gathering preference comparisons between trajectory fragments."""

    def __init__(
        self,
        seed: Optional[int] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Initializes the preference gatherer.

        Args:
            seed: seed for the internal RNG, if applicable
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        # The random seed isn't used here, but it's useful to have this
        # as an argument nevertheless because that means we can always
        # pass in a seed in training scripts (without worrying about whether
        # the PreferenceGatherer we use needs one).
        del seed
        self.logger = custom_logger or imit_logger.configure()

    @abc.abstractmethod
    def __call__(self, fragment_pairs: Sequence[TrajectoryWithRewPair]) -> np.ndarray:
        """Gathers the probabilities that fragment 1 is preferred in `fragment_pairs`.

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
        """  # noqa: DAR202


class SyntheticGatherer(PreferenceGatherer):
    """Computes synthetic preferences using ground-truth environment rewards."""

    def __init__(
        self,
        temperature: float = 1,
        discount_factor: float = 1,
        sample: bool = True,
        seed: Optional[int] = None,
        threshold: float = 50,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Initialize the synthetic preference gatherer.

        Args:
            temperature: the preferences are sampled from a softmax, this is
                the temperature used for sampling. temperature=0 leads to deterministic
                results (for equal rewards, 0.5 will be returned).
            discount_factor: discount factor that is used to compute
                how good a fragment is. Default is to use undiscounted
                sums of rewards (as in the DRLHP paper).
            sample: if True (default), the preferences are 0 or 1, sampled from
                a Bernoulli distribution (or 0.5 in the case of ties with zero
                temperature). If False, then the underlying Bernoulli probabilities
                are returned instead.
            seed: seed for the internal RNG (only used if temperature > 0 and sample)
            threshold: preferences are sampled from a softmax of returns.
                To avoid overflows, we clip differences in returns that are
                above this threshold (after multiplying with temperature).
                This threshold is therefore in logspace. The default value
                of 50 means that probabilities below 2e-22 are rounded up to 2e-22.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        super().__init__(custom_logger=custom_logger)
        self.temperature = temperature
        self.discount_factor = discount_factor
        self.sample = sample
        self.rng = np.random.default_rng(seed=seed)
        self.threshold = threshold

    def __call__(self, fragment_pairs: Sequence[TrajectoryWithRewPair]) -> np.ndarray:
        """Computes probability fragment 1 is preferred over fragment 2."""
        returns1, returns2 = self._reward_sums(fragment_pairs)
        if self.temperature == 0:
            return (np.sign(returns1 - returns2) + 1) / 2

        returns1 /= self.temperature
        returns2 /= self.temperature

        # clip the returns to avoid overflows in the softmax below
        returns_diff = np.clip(returns2 - returns1, -self.threshold, self.threshold)
        # Instead of computing exp(rews1) / (exp(rews1) + exp(rews2)) directly,
        # we divide enumerator and denominator by exp(rews1) to prevent overflows:
        model_probs = 1 / (1 + np.exp(returns_diff))
        # Compute the mean binary entropy. This metric helps estimate
        # how good we can expect the performance of the learned reward
        # model to be at predicting preferences.
        entropy = -(
            special.xlogy(model_probs, model_probs)
            + special.xlogy(1 - model_probs, 1 - model_probs)
        ).mean()
        self.logger.record("entropy", entropy)

        if self.sample:
            return self.rng.binomial(n=1, p=model_probs).astype(np.float32)
        return model_probs

    def _reward_sums(self, fragment_pairs) -> Tuple[np.ndarray, np.ndarray]:
        rews1, rews2 = zip(
            *[
                (
                    rollout.discounted_sum(f1.rews, self.discount_factor),
                    rollout.discounted_sum(f2.rews, self.discount_factor),
                )
                for f1, f2 in fragment_pairs
            ],
        )
        return np.array(rews1, dtype=np.float32), np.array(rews2, dtype=np.float32)


class PreferenceDataset(th.utils.data.Dataset):
    """A PyTorch Dataset for preference comparisons.

    Each item is a tuple consisting of two trajectory fragments
    and a probability that fragment 1 is preferred over fragment 2.

    This dataset is meant to be generated piece by piece during the
    training process, which is why data can be added via the .push()
    method.
    """

    def __init__(self):
        """Builds an empty PreferenceDataset."""
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

        Raises:
            ValueError: `preferences` shape does not match `fragments` or
                has non-float32 dtype.
        """
        fragments1, fragments2 = zip(*fragments)
        if preferences.shape != (len(fragments),):
            raise ValueError(
                f"Unexpected preferences shape {preferences.shape}, "
                f"expected {(len(fragments), )}",
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

    def save(self, path: AnyPath) -> None:
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: AnyPath) -> "PreferenceDataset":
        with open(path, "rb") as file:
            return pickle.load(file)


def preference_collate_fn(
    batch: Sequence[Tuple[TrajectoryWithRewPair, float]],
) -> Tuple[Sequence[TrajectoryWithRewPair], np.ndarray]:
    fragment_pairs, preferences = zip(*batch)
    return list(fragment_pairs), np.array(preferences)


class RewardTrainer(abc.ABC):
    """Abstract base class for training reward models using preference comparisons.

    This class contains only the actual reward model training code,
    it is not responsible for gathering trajectories and preferences
    or for agent training (see PreferenceComparisons for that).
    """

    def __init__(
        self,
        model: reward_nets.RewardNet,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Initialize the reward trainer.

        Args:
            model: the RewardNet instance to be trained
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self.model = model
        self.logger = custom_logger or imit_logger.configure()

    @abc.abstractmethod
    def train(self, dataset: PreferenceDataset, epoch_multiplier: float = 1.0):
        """Train the reward model on a batch of fragment pairs and preferences.

        Args:
            dataset: the dataset of preference comparisons to train on.
            epoch_multiplier: how much longer to train for than usual
                (measured relatively).
        """


class CrossEntropyRewardTrainer(RewardTrainer):
    """Train a reward model using a cross entropy loss."""

    def __init__(
        self,
        model: reward_nets.RewardNet,
        noise_prob: float = 0.0,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 1e-3,
        discount_factor: float = 1.0,
        threshold: float = 50,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Initialize the reward model trainer.

        Args:
            model: the RewardNet instance to be trained
            noise_prob: assumed probability with which the preference
                is uniformly random (used for the model of preference generation
                that is used for the loss)
            batch_size: number of fragment pairs per batch
            epochs: number of epochs on each training iteration (can be adjusted
                on the fly by specifying an `epoch_multiplier` in `self.train()`
                if longer training is desired in specific cases).
            lr: the learning rate
            discount_factor: the model of preference generation uses a softmax
                of returns as the probability that a fragment is preferred.
                This is the discount factor used to calculate those returns.
                Default is 1, i.e. undiscounted sums of rewards (which is what
                the DRLHP paper uses).
            threshold: the preference model used to compute the loss contains
                a softmax of returns. To avoid overflows, we clip differences
                in returns that are above this threshold. This threshold
                is therefore in logspace. The default value of 50 means
                that probabilities below 2e-22 are rounded up to 2e-22.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        super().__init__(model, custom_logger)
        self.noise_prob = noise_prob
        self.batch_size = batch_size
        self.epochs = epochs
        self.discount_factor = discount_factor
        self.threshold = threshold
        self.optim = th.optim.Adam(self.model.parameters(), lr=lr)

    def _loss(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
    ) -> th.Tensor:
        """Computes the loss.

        Args:
            fragment_pairs: Batch consisting of pairs of trajectory fragments.
            preferences: The probability that the first fragment is preferred
                over the second. Typically 0, 1 or 0.5 (tie).

        Returns:
            The cross-entropy loss between the probability predicted by the
            reward model and the target probabilities in `preferences`.
        """
        probs = th.empty(len(fragment_pairs), dtype=th.float32)
        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])
            rews1 = self._rewards(trans1)
            rews2 = self._rewards(trans2)
            probs[i] = self._probability(rews1, rews2)
        # TODO(ejnnr): Here and below, > 0.5 is problematic
        # because getting exactly 0.5 is actually somewhat
        # common in some environments (as long as sample=False or temperature=0).
        # In a sense that "only" creates class imbalance
        # but it's still misleading.
        predictions = (probs > 0.5).float()
        preferences_th = th.as_tensor(preferences, dtype=th.float32)
        ground_truth = (preferences_th > 0.5).float()
        accuracy = (predictions == ground_truth).float().mean()
        self.logger.record("accuracy", accuracy.item())
        return th.nn.functional.binary_cross_entropy(probs, preferences_th)

    def _rewards(self, transitions: Transitions) -> th.Tensor:
        preprocessed = self.model.preprocess(
            state=transitions.obs,
            action=transitions.acts,
            next_state=transitions.next_obs,
            done=transitions.dones,
        )
        return self.model(*preprocessed)

    def _probability(self, rews1: th.Tensor, rews2: th.Tensor) -> th.Tensor:
        """Computes the Boltzmann rational probability that the first trajectory is best.

        Args:
            rews1: A 1-dimensional array of rewards for the first trajectory fragment.
            rews2: A 1-dimensional array of rewards for the second trajectory fragment.

        Returns:
            The softmax of the difference between the (discounted) return of the
            first and second trajectory.
        """
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
        # Clip to avoid overflows (which in particular may occur
        # in the backwards pass even if they do not in the forward pass).
        returns_diff = th.clip(returns_diff, -self.threshold, self.threshold)
        # We take the softmax of the returns. model_probability
        # is the first dimension of that softmax, representing the
        # probability that fragment 1 is preferred.
        model_probability = 1 / (1 + returns_diff.exp())
        return self.noise_prob * 0.5 + (1 - self.noise_prob) * model_probability

    def train(self, dataset: PreferenceDataset, epoch_multiplier: float = 1.0):
        """Trains for `epoch_multiplier * self.epochs` epochs over `dataset`."""
        # TODO(ejnnr): This isn't specific to the loss function or probability model.
        # In general, it might be best to split the probability model, the loss and
        # the optimization procedure a bit more cleanly so that different versions
        # can be combined

        dataloader = th.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=preference_collate_fn,
        )
        epochs = round(self.epochs * epoch_multiplier)
        for _ in range(epochs):
            for fragment_pairs, preferences in dataloader:
                self.optim.zero_grad()
                loss = self._loss(fragment_pairs, preferences)
                loss.backward()
                self.optim.step()
                self.logger.record("loss", loss.item())


class PreferenceComparisons(base.BaseImitationAlgorithm):
    """Main interface for reward learning using preference comparisons."""

    def __init__(
        self,
        trajectory_generator: TrajectoryGenerator,
        reward_model: reward_nets.RewardNet,
        fragmenter: Optional[Fragmenter] = None,
        preference_gatherer: Optional[PreferenceGatherer] = None,
        reward_trainer: Optional[RewardTrainer] = None,
        comparisons_per_iteration: int = 50,
        fragment_length: int = 50,
        transition_oversampling: float = 10,
        initial_comparison_frac: float = 0.1,
        initial_epoch_multiplier: float = 200.0,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False,
        seed: Optional[int] = None,
    ):
        """Initialize the preference comparison trainer.

        The loggers of all subcomponents are overridden with the logger used
        by this class.

        Args:
            trajectory_generator: generates trajectories while optionally training
                an RL agent on the learned reward function (can also be a sampler
                from a static dataset of trajectories though).
            reward_model: a RewardNet instance to be used for learning the reward
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
            comparisons_per_iteration: number of preferences to gather at once (before
                switching back to agent training). This doesn't impact the total number
                of comparisons that are gathered, only the frequency of switching
                between preference gathering and agent training.
            fragment_length: number of timesteps per fragment that is used to elicit
                preferences
            transition_oversampling: factor by which to oversample transitions before
                creating fragments. Since fragments are sampled with replacement,
                this is usually chosen > 1 to avoid having the same transition
                in too many fragments.
            initial_comparison_frac: fraction of the total_comparisons argument
                to train() that will be sampled before the rest of training begins
                (using a randomly initialized agent). This can be used to pretrain the
                reward model before the agent is trained on the learned reward, to
                help avoid irreversibly learning a bad policy from an untrained reward.
                Note that there will often be some additional pretraining comparisons
                since `comparisons_per_iteration` won't exactly divide the total number
                of comparisons. How many such comparisons there are depends
                discontinuously on `total_comparisons` and `comparisons_per_iteration`.
            initial_epoch_multiplier: before agent training begins, train the reward
                model for this many more epochs than usual (on fragments sampled from a
                random agent).
            custom_logger: Where to log to; if None (default), creates a new logger.
            allow_variable_horizon: If False (default), algorithm will raise an
                exception if it detects trajectories of different length during
                training. If True, overrides this safety check. WARNING: variable
                horizon episodes leak information about the reward via termination
                condition, and can seriously confound evaluation. Read
                https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
                before overriding this.
            seed: seed to use for initializing subcomponents such as fragmenter.
                Only used when default components are used; if you instantiate your
                own fragmenter, preference gatherer, etc., you are responsible for
                seeding them!
        """
        super().__init__(
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )

        # for keeping track of the global iteration, in case train() is called
        # multiple times
        self._iteration = 0

        self.model = reward_model
        self.reward_trainer = reward_trainer or CrossEntropyRewardTrainer(
            reward_model,
            custom_logger=self.logger,
        )
        # If the reward trainer was created in the previous line, we've already passed
        # the correct logger. But if the user created a RewardTrainer themselves and
        # didn't manually set a logger, it would be annoying if a separate one was used.
        self.reward_trainer.logger = self.logger
        # the reward_trainer's model should refer to the same object as our copy
        assert self.reward_trainer.model is self.model
        self.trajectory_generator = trajectory_generator
        self.trajectory_generator.logger = self.logger
        self.fragmenter = fragmenter or RandomFragmenter(
            custom_logger=self.logger,
            seed=seed,
        )
        self.fragmenter.logger = self.logger
        self.preference_gatherer = preference_gatherer or SyntheticGatherer(
            custom_logger=self.logger,
            seed=seed,
        )
        self.preference_gatherer.logger = self.logger

        self.comparisons_per_iteration = comparisons_per_iteration
        self.fragment_length = fragment_length
        self.transition_oversampling = transition_oversampling
        self.initial_comparison_frac = initial_comparison_frac
        self.initial_epoch_multiplier = initial_epoch_multiplier

        self.dataset = PreferenceDataset()

    def train(
        self,
        total_timesteps: int,
        total_comparisons: int,
        callback: Optional[Callable[[int], None]] = None,
    ) -> Mapping[str, Any]:
        """Train the reward model and the policy if applicable.

        Args:
            total_timesteps: number of environment interaction steps
            total_comparisons: number of preferences to gather in total
            callback: callback functions called at the end of each iteration

        Returns:
            A dictionary with final metrics such as loss and accuracy
            of the reward model.

        Raises:
            ValueError: `total_comparisons < self.comparisons_per_iteration`.
        """
        initial_comparisons = int(total_comparisons * self.initial_comparison_frac)
        total_comparisons -= initial_comparisons
        iterations, extra_comparisons = divmod(
            total_comparisons,
            self.comparisons_per_iteration,
        )
        if iterations == 0:
            raise ValueError(
                f"total_comparisons={total_comparisons} is less than "
                f"comparisons_per_iteration={self.comparisons_per_iteration}",
            )
        timesteps_per_iteration, extra_timesteps = divmod(total_timesteps, iterations)

        reward_loss = None
        reward_accuracy = None

        for i in range(iterations):
            ##########################
            # Gather new preferences #
            ##########################
            num_pairs = self.comparisons_per_iteration
            # If the number of comparisons per iterations doesn't exactly divide
            # the desired total number of comparisons, we collect the remainder
            # right at the beginning to pretrain the reward model slightly.
            # WARNING: This means that slightly changing the total number of
            # comparisons or the number of comparisons per iteration can
            # significantly change the proportion of pretraining comparisons!
            #
            # In addition, we collect the comparisons specified via
            # initial_comparison_frac.
            if i == 0:
                num_pairs += extra_comparisons + initial_comparisons
            num_steps = math.ceil(
                self.transition_oversampling * 2 * num_pairs * self.fragment_length,
            )
            self.logger.log(f"Collecting {num_steps} trajectory steps")
            trajectories = self.trajectory_generator.sample(num_steps)
            # This assumes there are no fragments missing initial timesteps
            # (but allows for fragments missing terminal timesteps).
            horizons = (len(traj) for traj in trajectories if traj.terminal)
            self._check_fixed_horizon(horizons)
            self.logger.log("Creating fragment pairs")
            fragments = self.fragmenter(trajectories, self.fragment_length, num_pairs)
            with self.logger.accumulate_means("preferences"):
                self.logger.log("gathering preferences")
                preferences = self.preference_gatherer(fragments)
            self.dataset.push(fragments, preferences)
            self.logger.log(f"Dataset now contains {len(self.dataset)} samples")

            ##########################
            # Train the reward model #
            ##########################

            # On the first iteration, we train the reward model for longer,
            # as specified by initial_epoch_multiplier.
            epoch_multiplier = 1.0
            if i == 0:
                epoch_multiplier = self.initial_epoch_multiplier

            with self.logger.accumulate_means("reward"):
                self.logger.log("Training reward model")
                self.reward_trainer.train(
                    self.dataset,
                    epoch_multiplier=epoch_multiplier,
                )
            reward_loss = self.logger.name_to_value["mean/reward/loss"]
            reward_accuracy = self.logger.name_to_value["mean/reward/accuracy"]

            ###################
            # Train the agent #
            ###################
            num_steps = timesteps_per_iteration
            # if the number of timesteps per iterations doesn't exactly divide
            # the desired total number of timesteps, we train the agent a bit longer
            # at the end of training (where the reward model is presumably best)
            if i == iterations - 1:
                num_steps += extra_timesteps
            with self.logger.accumulate_means("agent"):
                self.logger.log(f"Training agent for {num_steps} timesteps")
                self.trajectory_generator.train(steps=num_steps)

            self.logger.dump(self._iteration)

            ########################
            # Additional Callbacks #
            ########################
            if callback:
                callback(self._iteration)
            self._iteration += 1

        return {"reward_loss": reward_loss, "reward_accuracy": reward_accuracy}
