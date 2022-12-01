"""Type alias shared by reward-related code."""

import abc
from typing import Protocol

import numpy as np

import imitation.policies.replay_buffer_wrapper


class RewardFn(Protocol):
    """Abstract class for reward function.

    Requires implementation of __call__() to compute the reward given a batch of
    states, actions, next states and dones.
    """

    @abc.abstractmethod
    def __call__(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Compute rewards for a batch of transitions.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.

        Returns:
            Computed rewards of shape `(batch_size,`).
        """  # noqa: DAR202


class ReplayBufferAwareRewardFn(RewardFn, abc.ABC):
    """Abstract class for a reward function that needs access to a replay buffer."""

    @abc.abstractmethod
    def on_replay_buffer_initialized(
        self,
        replay_buffer: (
            "imitation.policies.replay_buffer_wrapper.ReplayBufferRewardWrapper"
        ),
    ):
        pass
