"""Learning reward models using preference comparisons.

Trains a reward model and optionally a policy based on preferences
between trajectory fragments.
"""
import abc
import math
import pickle
import random
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch as th
from scipy import special
from stable_baselines3.common import base_class, type_aliases, utils, vec_env
from torch import nn
from torch.utils import data as data_th
from tqdm.auto import tqdm

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
from imitation.rewards import reward_function, reward_nets, reward_wrapper
from imitation.util import logger as imit_logger
from imitation.util import networks, util


class TrajectoryGenerator(abc.ABC):
    """Generator of trajectories with optional training logic."""

    _logger: imit_logger.HierarchicalLogger
    """Object to log statistics and natural language messages to."""

    def __init__(self, custom_logger: Optional[imit_logger.HierarchicalLogger] = None):
        """Builds TrajectoryGenerator.

        Args:
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self.logger = custom_logger or imit_logger.configure()

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
        trajectories: Sequence[TrajectoryWithRew],
        seed: Optional[int] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Creates a dataset loaded from `path`.

        Args:
            trajectories: the dataset of rollouts.
            seed: Seed for RNG used for shuffling dataset.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        super().__init__(custom_logger=custom_logger)
        self._trajectories = trajectories
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
        reward_fn: Union[reward_function.RewardFn, reward_nets.RewardNet],
        venv: vec_env.VecEnv,
        exploration_frac: float = 0.0,
        switch_prob: float = 0.5,
        random_prob: float = 0.5,
        seed: Optional[int] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Initialize the agent trainer.

        Args:
            algorithm: the stable-baselines algorithm to use for training.
            reward_fn: either a RewardFn or a RewardNet instance that will supply
                the rewards used for training the agent.
            venv: vectorized environment to train in.
            exploration_frac: fraction of the trajectories that will be generated
                partially randomly rather than only by the agent when sampling.
            switch_prob: the probability of switching the current policy at each
                step for the exploratory samples.
            random_prob: the probability of picking the random policy when switching
                during exploration.
            seed: random seed for exploratory trajectories.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self.algorithm = algorithm
        # NOTE: this has to come after setting self.algorithm because super().__init__
        # will set self.logger, which also sets the logger for the algorithm
        super().__init__(custom_logger)
        if isinstance(reward_fn, reward_nets.RewardNet):
            utils.check_for_correct_spaces(
                venv,
                reward_fn.observation_space,
                reward_fn.action_space,
            )
            reward_fn = reward_fn.predict_processed
        self.reward_fn = reward_fn
        self.exploration_frac = exploration_frac

        # The BufferingWrapper records all trajectories, so we can return
        # them after training. This should come first (before the wrapper that
        # changes the reward function), so that we return the original environment
        # rewards.
        # When applying BufferingWrapper and RewardVecEnvWrapper, we should use `venv`
        # instead of `algorithm.get_env()` because SB3 may apply some wrappers to
        # `algorithm`'s env under the hood. In particular, in image-based environments,
        # SB3 may move the image-channel dimension in the observation space, making
        # `algorithm.get_env()` not match with `reward_fn`.
        self.buffering_wrapper = wrappers.BufferingWrapper(venv)
        self.venv = self.reward_venv_wrapper = reward_wrapper.RewardVecEnvWrapper(
            self.buffering_wrapper,
            reward_fn=self.reward_fn,
        )

        self.log_callback = self.reward_venv_wrapper.make_log_callback()

        self.algorithm.set_env(self.venv)
        # Unlike with BufferingWrapper, we should use `algorithm.get_env()` instead
        # of `venv` when interacting with `algorithm`.
        policy_callable = rollout._policy_to_callable(
            self.algorithm,
            self.algorithm.get_env(),
            # By setting deterministic_policy to False, we ensure that the rollouts
            # are collected from a deterministic policy only if self.algorithm is
            # deterministic. If self.algorithm is stochastic, then policy_callable
            # will also be stochastic.
            deterministic_policy=False,
        )
        self.exploration_wrapper = exploration_wrapper.ExplorationWrapper(
            policy_callable=policy_callable,
            venv=self.algorithm.get_env(),
            random_prob=random_prob,
            switch_prob=switch_prob,
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
        self.algorithm.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=self.log_callback,
            **kwargs,
        )

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
                self.algorithm.get_env(),
                sample_until=sample_until,
                # By setting deterministic_policy to False, we ensure that the rollouts
                # are collected from a deterministic policy only if self.algorithm is
                # deterministic. If self.algorithm is stochastic, then policy_callable
                # will also be stochastic.
                deterministic_policy=False,
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
                venv=self.algorithm.get_env(),
                sample_until=sample_until,
                # buffering_wrapper collects rollouts from a non-deterministic policy
                # so we do that here as well for consistency.
                deterministic_policy=False,
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

    def __init__(self, max_size: Optional[int] = None):
        """Builds an empty PreferenceDataset.

        Args:
            max_size: Maximum number of preference comparisons to store in the dataset.
                If None (default), the dataset can grow indefinitely. Otherwise, the
                dataset acts as a FIFO queue, and the oldest comparisons are evicted
                when `push()` is called and the dataset is at max capacity.
        """
        self.fragments1: List[TrajectoryWithRew] = []
        self.fragments2: List[TrajectoryWithRew] = []
        self.max_size = max_size
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

        # Evict old samples if the dataset is at max capacity
        if self.max_size is not None:
            extra = len(self.preferences) - self.max_size
            if extra > 0:
                self.fragments1 = self.fragments1[extra:]
                self.fragments2 = self.fragments2[extra:]
                self.preferences = self.preferences[extra:]

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


class LossAndMetrics(NamedTuple):
    """Loss and auxiliary metrics for reward network training."""

    loss: th.Tensor
    metrics: Mapping[str, th.Tensor]


class RewardLoss(nn.Module, abc.ABC):
    """A loss function over preferences."""

    @abc.abstractmethod
    def forward(
        self,
        model: reward_nets.RewardNet,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
    ) -> LossAndMetrics:
        """Computes the loss.

        Args:
            model: the reward network
            fragment_pairs: Batch consisting of pairs of trajectory fragments.
            preferences: The probability that the first fragment is preferred
                over the second. Typically 0, 1 or 0.5 (tie).

        Returns: # noqa: DAR202
            loss: the loss
            metrics: a dictionary of metrics that can be logged
        """


def _evaluate_reward_on_transitions(
    model: reward_nets.RewardNet,
    transitions: Transitions,
) -> th.Tensor:
    """Evaluate `model` on `transitions` and return the rewards with gradients."""
    preprocessed = model.preprocess(
        state=transitions.obs,
        action=transitions.acts,
        next_state=transitions.next_obs,
        done=transitions.dones,
    )
    return model(*preprocessed)


def _trajectory_pair_includes_reward(fragment_pair: TrajectoryPair):
    """Return true if and only if both fragments in the pair include rewards."""
    frag1, frag2 = fragment_pair
    return isinstance(frag1, TrajectoryWithRew) and isinstance(frag2, TrajectoryWithRew)


class CrossEntropyRewardLoss(RewardLoss):
    """Compute the cross entropy reward loss."""

    def __init__(
        self,
        noise_prob: float = 0.0,
        discount_factor: float = 1.0,
        threshold: float = 50,
    ):
        """Create cross entropy reward loss.

        Args:
            noise_prob: assumed probability with which the preference
                is uniformly random (used for the model of preference generation
                that is used for the loss)
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
        """
        super().__init__()
        self.discount_factor = discount_factor
        self.noise_prob = noise_prob
        self.threshold = threshold

    def forward(
        self,
        model: reward_nets.RewardNet,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
    ) -> LossAndMetrics:
        """Computes the loss.

        Args:
            model: the reward network to call
            fragment_pairs: Batch consisting of pairs of trajectory fragments.
            preferences: The probability that the first fragment is preferred
                over the second. Typically 0, 1 or 0.5 (tie).

        Returns:
            The cross-entropy loss between the probability predicted by the
                reward model and the target probabilities in `preferences`. Metrics
                are accuracy, and gt_reward_loss, if the ground truth reward is
                available.
        """
        gt_reward_available = _trajectory_pair_includes_reward(fragment_pairs[0])
        probs = th.empty(len(fragment_pairs), dtype=th.float32)
        if gt_reward_available:
            gt_probs = th.empty(len(fragment_pairs), dtype=th.float32)

        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])
            rews1 = _evaluate_reward_on_transitions(model, trans1)
            rews2 = _evaluate_reward_on_transitions(model, trans2)
            probs[i] = self._probability(rews1, rews2)
            if gt_reward_available:
                gt_rews_1 = th.from_numpy(frag1.rews)
                gt_rews_2 = th.from_numpy(frag2.rews)
                gt_probs[i] = self._probability(gt_rews_1, gt_rews_2)
        # TODO(ejnnr): Here and below, > 0.5 is problematic
        # because getting exactly 0.5 is actually somewhat
        # common in some environments (as long as sample=False or temperature=0).
        # In a sense that "only" creates class imbalance
        # but it's still misleading.
        predictions = (probs > 0.5).float()
        preferences_th = th.as_tensor(preferences, dtype=th.float32)
        ground_truth = (preferences_th > 0.5).float()
        metrics = {}
        metrics["accuracy"] = (predictions == ground_truth).float().mean()
        if gt_reward_available:
            metrics["gt_reward_loss"] = th.nn.functional.binary_cross_entropy(
                gt_probs,
                preferences_th,
            )
        metrics = {key: value.detach().cpu() for key, value in metrics.items()}
        return LossAndMetrics(
            loss=th.nn.functional.binary_cross_entropy(probs, preferences_th),
            metrics=metrics,
        )

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


class RewardTrainer(abc.ABC):
    """Abstract base class for training reward models using preference comparisons.

    This class contains only the actual reward model training code,
    it is not responsible for gathering trajectories and preferences
    or for agent training (see :class: `PreferenceComparisons` for that).
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
        self._model = model
        self.logger = custom_logger or imit_logger.configure()

    def train(self, dataset: PreferenceDataset, epoch_multiplier: float = 1.0) -> None:
        """Train the reward model on a batch of fragment pairs and preferences.

        Args:
            dataset: the dataset of preference comparisons to train on.
            epoch_multiplier: how much longer to train for than usual
                (measured relatively).
        """
        with networks.training(self._model):
            self._train(dataset, epoch_multiplier)

    @abc.abstractmethod
    def _train(self, dataset: PreferenceDataset, epoch_multiplier: float) -> None:
        """Train the reward model; see ``train`` for details."""


class BasicRewardTrainer(RewardTrainer):
    """Train a basic reward model."""

    def __init__(
        self,
        model: reward_nets.RewardNet,
        loss: RewardLoss,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Initialize the reward model trainer.

        Args:
            model: the RewardNet instance to be trained
            loss: the loss to use
            batch_size: number of fragment pairs per batch
            epochs: number of epochs in each training iteration (can be adjusted
                on the fly by specifying an `epoch_multiplier` in `self.train()`
                if longer training is desired in specific cases).
            lr: the learning rate
            weight_decay: the weight decay factor for the reward model's weights
                to use with ``th.optim.AdamW``. This is similar to but not equivalent
                to L2 regularization, see https://arxiv.org/abs/1711.05101
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        super().__init__(model, custom_logger)
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.optim = th.optim.AdamW(
            self._model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def _make_data_loader(self, dataset: PreferenceDataset) -> data_th.DataLoader:
        """Make a dataloader."""
        return data_th.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=preference_collate_fn,
        )

    def _train(self, dataset: PreferenceDataset, epoch_multiplier: float = 1.0) -> None:
        """Trains for `epoch_multiplier * self.epochs` epochs over `dataset`."""
        dataloader = self._make_data_loader(dataset)
        epochs = round(self.epochs * epoch_multiplier)

        for _ in tqdm(range(epochs), desc="Training reward model"):
            for fragment_pairs, preferences in dataloader:
                self.optim.zero_grad()
                loss = self._training_inner_loop(fragment_pairs, preferences)
                loss.backward()
                self.optim.step()

    def _training_inner_loop(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
    ) -> th.Tensor:
        output = self.loss.forward(self._model, fragment_pairs, preferences)
        loss = output.loss
        self.logger.record("loss", loss.item())
        for name, value in output.metrics.items():
            self.logger.record(name, value.item())
        return loss


class EnsembleTrainer(BasicRewardTrainer):
    """Train a reward ensemble."""

    _model: reward_nets.RewardEnsemble

    def __init__(
        self,
        model: reward_nets.RewardEnsemble,
        loss: RewardLoss,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Initialize the reward model trainer.

        Args:
            model: the RewardNet instance to be trained
            loss: the loss to use
            batch_size: number of fragment pairs per batch
            epochs: number of epochs in each training iteration (can be adjusted
                on the fly by specifying an `epoch_multiplier` in `self.train()`
                if longer training is desired in specific cases).
            lr: the learning rate
            weight_decay: the weight decay factor for the reward model's weights
                to use with ``th.optim.AdamW``. This is similar to but not equivalent
                to L2 regularization, see https://arxiv.org/abs/1711.05101
            custom_logger: Where to log to; if None (default), creates a new logger.

        Raises:
            TypeError: if model is not a RewardEnsemble.
        """
        if not isinstance(model, reward_nets.RewardEnsemble):
            raise TypeError(
                f"RewardEnsemble expected by EnsembleTrainer not {type(model)}.",
            )

        super().__init__(
            model,
            loss,
            batch_size,
            epochs,
            lr,
            weight_decay,
            custom_logger,
        )

    def _training_inner_loop(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
    ) -> th.Tensor:
        losses = []
        metrics = []
        for member in self._model.members:
            output = self.loss.forward(member, fragment_pairs, preferences)
            losses.append(output.loss)
            metrics.append(output.metrics)
        losses = th.stack(losses)
        loss = losses.sum()

        self.logger.record("loss", loss.item())
        self.logger.record("loss_std", losses.std().item())

        # Turn metrics from a list of dictionaries into a dictionary of
        # tensors.
        metrics = {k: th.stack([di[k] for di in metrics]) for k in metrics[0]}
        for name, value in metrics.items():
            self.logger.record(name, value.mean().item())

        return loss


def _make_reward_trainer(
    reward_model: reward_nets.RewardNet,
    loss: RewardLoss,
    reward_trainer_kwargs: Optional[Mapping[str, Any]] = None,
) -> RewardTrainer:
    """Construct the correct type of reward trainer for this reward function."""
    if reward_trainer_kwargs is None:
        reward_trainer_kwargs = {}

    base_model = reward_model
    while hasattr(base_model, "base"):
        base_model = base_model.base

    if isinstance(base_model, reward_nets.RewardEnsemble):
        # reward_model may include an AddSTDRewardWrapper for RL training; but we
        # must train directly on the base model for reward model training.
        is_base = reward_model is base_model
        is_std_wrapper = (
            isinstance(reward_model, reward_nets.AddSTDRewardWrapper)
            and reward_model.base is base_model
        )

        if is_base or is_std_wrapper:
            return EnsembleTrainer(base_model, loss, **reward_trainer_kwargs)
        else:
            raise ValueError(
                "RewardEnsemble can only be wrapped"
                f" by AddSTDRewardWrapper but found {type(reward_model).__name__}.",
            )
    else:
        return BasicRewardTrainer(
            reward_model,
            loss=loss,
            **reward_trainer_kwargs,
        )


QUERY_SCHEDULES: Dict[str, type_aliases.Schedule] = {
    "constant": lambda t: 1.0,
    "hyperbolic": lambda t: 1.0 / (1.0 + t),
    "inverse_quadratic": lambda t: 1.0 / (1.0 + t**2),
}


class PreferenceComparisons(base.BaseImitationAlgorithm):
    """Main interface for reward learning using preference comparisons."""

    def __init__(
        self,
        trajectory_generator: TrajectoryGenerator,
        reward_model: reward_nets.RewardNet,
        num_iterations: int,
        fragmenter: Optional[Fragmenter] = None,
        preference_gatherer: Optional[PreferenceGatherer] = None,
        reward_trainer: Optional[RewardTrainer] = None,
        comparison_queue_size: Optional[int] = None,
        fragment_length: int = 100,
        transition_oversampling: float = 1,
        initial_comparison_frac: float = 0.1,
        initial_epoch_multiplier: float = 200.0,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False,
        seed: Optional[int] = None,
        query_schedule: Union[str, type_aliases.Schedule] = "hyperbolic",
    ):
        """Initialize the preference comparison trainer.

        The loggers of all subcomponents are overridden with the logger used
        by this class.

        Args:
            trajectory_generator: generates trajectories while optionally training
                an RL agent on the learned reward function (can also be a sampler
                from a static dataset of trajectories though).
            reward_model: a RewardNet instance to be used for learning the reward
            num_iterations: number of times to train the agent against the reward model
                and then train the reward model against newly gathered preferences.
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
            comparison_queue_size: the maximum number of comparisons to keep in the
                queue for training the reward model. If None, the queue will grow
                without bound as new comparisons are added.
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
            query_schedule: one of ("constant", "hyperbolic", "inverse_quadratic"), or
                a function that takes in a float between 0 and 1 inclusive,
                representing a fraction of the total number of timesteps elapsed up to
                some time T, and returns a potentially unnormalized probability
                indicating the fraction of `total_comparisons` that should be queried
                at that iteration. This function will be called `num_iterations` times
                in `__init__()` with values from `np.linspace(0, 1, num_iterations)`
                as input. The outputs will be normalized to sum to 1 and then used to
                apportion the comparisons among the `num_iterations` iterations.

        Raises:
            ValueError: if `query_schedule` is not a valid string or callable.
        """
        super().__init__(
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )

        # for keeping track of the global iteration, in case train() is called
        # multiple times
        self._iteration = 0

        self.model = reward_model

        if reward_trainer is None:
            loss = CrossEntropyRewardLoss()
            self.reward_trainer = _make_reward_trainer(reward_model, loss)
        else:
            self.reward_trainer = reward_trainer

        # If the reward trainer was created in the previous line, we've already passed
        # the correct logger. But if the user created a RewardTrainer themselves and
        # didn't manually set a logger, it would be annoying if a separate one was used.
        self.reward_trainer.logger = self.logger
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

        self.fragment_length = fragment_length
        self.initial_comparison_frac = initial_comparison_frac
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.num_iterations = num_iterations
        self.transition_oversampling = transition_oversampling
        if callable(query_schedule):
            self.query_schedule = query_schedule
        elif query_schedule in QUERY_SCHEDULES:
            self.query_schedule = QUERY_SCHEDULES[query_schedule]
        else:
            raise ValueError(f"Unknown query schedule: {query_schedule}")

        self.dataset = PreferenceDataset(max_size=comparison_queue_size)

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
        """
        initial_comparisons = int(total_comparisons * self.initial_comparison_frac)
        total_comparisons -= initial_comparisons

        # Compute the number of comparisons to request at each iteration in advance.
        vec_schedule = np.vectorize(self.query_schedule)
        unnormalized_probs = vec_schedule(np.linspace(0, 1, self.num_iterations))
        probs = unnormalized_probs / np.sum(unnormalized_probs)
        shares = util.oric(probs * total_comparisons)
        schedule = [initial_comparisons] + shares.tolist()
        print(f"Query schedule: {schedule}")

        timesteps_per_iteration, extra_timesteps = divmod(
            total_timesteps,
            self.num_iterations,
        )
        reward_loss = None
        reward_accuracy = None

        for i, num_pairs in enumerate(schedule):
            ##########################
            # Gather new preferences #
            ##########################
            num_steps = math.ceil(
                self.transition_oversampling * 2 * num_pairs * self.fragment_length,
            )
            self.logger.log(
                f"Collecting {2 * num_pairs} fragments ({num_steps} transitions)",
            )
            trajectories = self.trajectory_generator.sample(num_steps)
            # This assumes there are no fragments missing initial timesteps
            # (but allows for fragments missing terminal timesteps).
            horizons = (len(traj) for traj in trajectories if traj.terminal)
            self._check_fixed_horizon(horizons)
            self.logger.log("Creating fragment pairs")
            fragments = self.fragmenter(trajectories, self.fragment_length, num_pairs)
            with self.logger.accumulate_means("preferences"):
                self.logger.log("Gathering preferences")
                preferences = self.preference_gatherer(fragments)
            self.dataset.push(fragments, preferences)
            self.logger.log(f"Dataset now contains {len(self.dataset)} comparisons")

            ##########################
            # Train the reward model #
            ##########################

            # On the first iteration, we train the reward model for longer,
            # as specified by initial_epoch_multiplier.
            epoch_multiplier = 1.0
            if i == 0:
                epoch_multiplier = self.initial_epoch_multiplier

            with self.logger.accumulate_means("reward"):
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
            if i == self.num_iterations - 1:
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
