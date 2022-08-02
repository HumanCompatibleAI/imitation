"""Wrapper for reward labeling for transitions sampled from a replay buffer."""


from typing import Mapping, Optional, Type

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from imitation.rewards.reward_function import RewardFn
from imitation.util import util


def _samples_to_reward_fn_input(
    samples: ReplayBufferSamples,
) -> Mapping[str, np.ndarray]:
    """Convert a sample from a replay buffer to a numpy array."""
    return dict(
        state=samples.observations.cpu().numpy(),
        action=samples.actions.cpu().numpy(),
        next_state=samples.next_observations.cpu().numpy(),
        done=samples.dones.cpu().numpy(),
    )


class ReplayBufferRewardWrapper(ReplayBuffer):
    """Relabel the rewards in transitions sampled from a ReplayBuffer."""

    def __init__(
        self,
        *args,
        replay_buffer_cls: Type[ReplayBuffer] = ReplayBuffer,
        reward_fn: Optional[RewardFn] = None,
        **kwargs,
    ):
        """Builds ReplayBufferRewardWrapper.

        Note(yawen): we directly inherit ReplayBuffer in this case and leave out the
        choice of DictReplayBuffer because the current RewardFn only takes in NumPy
        array-based inputs, and SAC is the only use case for ReplayBuffer relabeling.

        Args:
            *args: Arguments to ReplayBuffer.
            replay_buffer_cls: Class of the replay buffer.
            reward_fn: Reward function for reward relabeling.
            **kwargs: keyword arguments for ReplayBuffer.
        """
        self.replay_buffer = replay_buffer_cls(*args, **kwargs)
        self.reward_fn = reward_fn

    def sample(self, *args, **kwargs):
        samples = self.replay_buffer.sample(*args, **kwargs)
        if self.reward_fn is None:
            return samples

        rewards = self.reward_fn(**_samples_to_reward_fn_input(samples))
        shape = samples.rewards.shape
        device = samples.rewards.device
        rewards_th = util.safe_to_tensor(rewards).reshape(shape).to(device)

        return ReplayBufferSamples(
            samples.observations,
            samples.actions,
            samples.next_observations,
            samples.dones,
            rewards_th,
        )

    def add(self, *args, **kwargs):
        self.replay_buffer.add(*args, **kwargs)

    def size(self) -> int:
        return self.replay_buffer.size()

    def extend(self, *args, **kwargs) -> None:
        self.replay_buffer.extend(*args, **kwargs)

    def reset(self) -> None:
        self.replay_buffer.reset()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        return self.replay_buffer.to_torch(array, copy)
