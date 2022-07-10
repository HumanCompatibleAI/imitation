"""Wrapper for reward labeling for transitions sampled from a replay buffer."""


import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from imitation.rewards.common import RewardFn


class ReplayBufferRewardWrapper(ReplayBuffer):
    def __init__(self, replay_buffer: ReplayBuffer, reward_fn: RewardFn):
        self.replay_buffer = replay_buffer
        self.reward_fn = reward_fn

    def sample(self, *args, **kwargs):
        samples = self.replay_buffer.sample(*args, **kwargs)
        rewards = self.reward_fn(
            samples.observations,
            samples.actions,
            samples.next_observations,
            samples.dones,
        )

        return ReplayBufferSamples(
            samples.observations,
            samples.actions,
            samples.next_observations,
            samples.dones,
            rewards,
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
