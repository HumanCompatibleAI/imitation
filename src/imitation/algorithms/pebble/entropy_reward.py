"""Reward function for the PEBBLE training algorithm."""

from enum import Enum, auto
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch as th

from imitation.policies.replay_buffer_wrapper import (
    ReplayBufferRewardWrapper,
    ReplayBufferView,
)
from imitation.rewards.reward_function import ReplayBufferAwareRewardFn, RewardFn
from imitation.util import util
from imitation.util.networks import RunningNorm


class PebbleRewardPhase(Enum):
    """States representing different behaviors for PebbleStateEntropyReward."""

    UNSUPERVISED_EXPLORATION = auto()  # Entropy based reward
    POLICY_AND_REWARD_LEARNING = auto()  # Learned reward


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
    supplied with set_replay_buffer() or on_replay_buffer_initialized().
    To transition to the last phase, unsupervised_exploration_finish() needs
    to be called.
    """

    def __init__(
        self,
        learned_reward_fn: RewardFn,
        nearest_neighbor_k: int = 5,
    ):
        """Builds this class.

        Args:
            learned_reward_fn: The learned reward function used after unsupervised
                exploration is finished
            nearest_neighbor_k: Parameter for entropy computation (see
                compute_state_entropy())
        """
        self.learned_reward_fn = learned_reward_fn
        self.nearest_neighbor_k = nearest_neighbor_k
        self.entropy_stats = RunningNorm(1)
        self.state = PebbleRewardPhase.UNSUPERVISED_EXPLORATION

        # These two need to be set with set_replay_buffer():
        self.replay_buffer_view: Optional[ReplayBufferView] = None
        self.obs_shape: Union[Tuple[int, ...], Dict[str, Tuple[int, ...]], None] = None

    def on_replay_buffer_initialized(self, replay_buffer: ReplayBufferRewardWrapper):
        self.set_replay_buffer(replay_buffer.buffer_view, replay_buffer.obs_shape)

    def set_replay_buffer(
        self,
        replay_buffer: ReplayBufferView,
        obs_shape: Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]],
    ):
        self.replay_buffer_view = replay_buffer
        self.obs_shape = obs_shape

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
            return self._entropy_reward(state, action, next_state, done)
        else:
            return self.learned_reward_fn(state, action, next_state, done)

    def _entropy_reward(self, state, action, next_state, done):
        if self.replay_buffer_view is None:
            raise ValueError(
                "Replay buffer must be supplied before entropy reward can be used",
            )
        all_observations = self.replay_buffer_view.observations
        # ReplayBuffer sampling flattens the venv dimension, let's adapt to that
        all_observations = all_observations.reshape((-1, *self.obs_shape))

        if all_observations.shape[0] < self.nearest_neighbor_k:
            # not enough observations to compare to, fall back to the learned function;
            # (falling back to a constant may also be ok)
            return self.learned_reward_fn(state, action, next_state, done)
        else:
            # TODO #625: deal with the conversion back and forth between np and torch
            entropies = util.compute_state_entropy(
                th.tensor(state),
                th.tensor(all_observations),
                self.nearest_neighbor_k,
            )
            normalized_entropies = self.entropy_stats.forward(entropies)
            return normalized_entropies.numpy()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["replay_buffer_view"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.replay_buffer_view = None
