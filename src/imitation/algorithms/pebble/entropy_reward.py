"""Reward function for the PEBBLE training algorithm."""

import enum
from typing import Any, Callable, Optional, Tuple

import gym
import numpy as np
import torch as th

from imitation.policies.replay_buffer_wrapper import (
    ReplayBufferAwareRewardFn,
    ReplayBufferRewardWrapper,
    ReplayBufferView,
)
from imitation.rewards.reward_function import RewardFn
from imitation.rewards.reward_nets import RewardNet
from imitation.util import util


class InsufficientObservations(RuntimeError):
    """Error signifying not enough observations for entropy calculation."""

    pass


class EntropyRewardNet(RewardNet, ReplayBufferAwareRewardFn):
    """RewardNet wrapping entropy reward function."""

    __call__: Callable[..., Any]  # Needed to appease pytype

    def __init__(
        self,
        nearest_neighbor_k: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        normalize_images: bool = True,
        replay_buffer_view: Optional[ReplayBufferView] = None,
    ):
        """Initialize the RewardNet.

        Args:
            nearest_neighbor_k: Parameter for entropy computation (see
                compute_state_entropy())
            observation_space: the observation space of the environment
            action_space: the action space of the environment
            normalize_images: whether to automatically normalize
                image observations to [0, 1] (from 0 to 255). Defaults to True.
            replay_buffer_view: Replay buffer view with observations to compare
                against when computing entropy. If None is given, the buffer needs to
                be set with on_replay_buffer_initialized() before EntropyRewardNet can
                be used
        """
        super().__init__(observation_space, action_space, normalize_images)
        self.nearest_neighbor_k = nearest_neighbor_k
        self._replay_buffer_view = replay_buffer_view

    def on_replay_buffer_initialized(self, replay_buffer: ReplayBufferRewardWrapper):
        """Sets replay buffer.

        This method needs to be called, e.g., after unpickling.
        See also __getstate__() / __setstate__().

        Args:
            replay_buffer: replay buffer with history of observations
        """
        assert self.observation_space == replay_buffer.observation_space
        assert self.action_space == replay_buffer.action_space
        self._replay_buffer_view = replay_buffer.buffer_view

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        assert (
            self._replay_buffer_view is not None
        ), "Missing replay buffer (possibly after unpickle)"

        all_observations = self._replay_buffer_view.observations
        # ReplayBuffer sampling flattens the venv dimension, let's adapt to that
        all_observations = all_observations.reshape(
            (-1,) + self.observation_space.shape,
        )

        if all_observations.shape[0] < self.nearest_neighbor_k:
            raise InsufficientObservations(
                "Insufficient observations for entropy calculation",
            )

        return util.compute_state_entropy(
            state,
            all_observations,
            self.nearest_neighbor_k,
        )

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Override default preprocessing to avoid the default one-hot encoding.

        We also know forward() only works with state, so no need to convert
        other tensors.

        Args:
            state: The observation input.
            action: The action input.
            next_state: The observation input.
            done: Whether the episode has terminated.

        Returns:
            Observations preprocessed by converting them to Tensor.
        """
        state_th = util.safe_to_tensor(state).to(self.device)
        action_th = next_state_th = done_th = th.empty(0)
        return state_th, action_th, next_state_th, done_th

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_replay_buffer_view"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._replay_buffer_view = None


class PebbleRewardPhase(enum.Enum):
    """States representing different behaviors for PebbleStateEntropyReward."""

    UNSUPERVISED_EXPLORATION = enum.auto()  # Entropy based reward
    POLICY_AND_REWARD_LEARNING = enum.auto()  # Learned reward


class PebbleStateEntropyReward(ReplayBufferAwareRewardFn):
    """Reward function for implementation of the PEBBLE learning algorithm.

    See https://arxiv.org/abs/2106.05091 .

    The rewards returned by this function go through the three phases:
    1. Before enough samples are collected for entropy calculation, the
    underlying function is returned. This shouldn't matter because
    OffPolicyAlgorithms have an initialization period for `learning_starts`
    timesteps.
    2. During the unsupervised exploration phase, entropy based reward is returned
    3. After unsupervised exploration phase is finished, the underlying learned
    reward is returned.

    The second phase requires that a buffer with observations to compare against is
    supplied with on_replay_buffer_initialized(). To transition to the last phase,
    unsupervised_exploration_finish() needs to be called.
    """

    def __init__(
        self,
        entropy_reward_fn: RewardFn,
        learned_reward_fn: RewardFn,
    ):
        """Builds this class.

        Args:
            entropy_reward_fn: The entropy-based reward function used during
                unsupervised exploration
            learned_reward_fn: The learned reward function used after unsupervised
                exploration is finished
        """
        self.entropy_reward_fn = entropy_reward_fn
        self.learned_reward_fn = learned_reward_fn
        self.state = PebbleRewardPhase.UNSUPERVISED_EXPLORATION

    def on_replay_buffer_initialized(self, replay_buffer: ReplayBufferRewardWrapper):
        if isinstance(self.entropy_reward_fn, ReplayBufferAwareRewardFn):
            self.entropy_reward_fn.on_replay_buffer_initialized(replay_buffer)

    def unsupervised_exploration_finish(self):
        assert self.state == PebbleRewardPhase.UNSUPERVISED_EXPLORATION
        self.state = PebbleRewardPhase.POLICY_AND_REWARD_LEARNING

    def __call__(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        if self.state == PebbleRewardPhase.UNSUPERVISED_EXPLORATION:
            try:
                return self.entropy_reward_fn(state, action, next_state, done)
            except InsufficientObservations:
                # not enough observations to compare to, fall back to the learned
                # function; (falling back to a constant may also be ok)
                return self.learned_reward_fn(state, action, next_state, done)
        else:
            return self.learned_reward_fn(state, action, next_state, done)
