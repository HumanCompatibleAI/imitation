"""Wrapper for reward labeling for transitions sampled from a replay buffer."""


from typing import Mapping, Union

import numpy as np
import torch as th
from gym import spaces
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
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        reward_fn: RewardFn,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        """Builds ReplayBufferRewardWrapper.

        Note(yawen): we directly inherit ReplayBuffer in this case and leave out the
        choice of DictReplayBuffer because the current RewardFn only takes in NumPy
        array-based inputs, and SAC is the only use case for ReplayBuffer relabeling.

        Args:
            buffer_size: Max number of element in the buffer
            observation_space: Observation space
            action_space: Action space
            reward_fn: The reward function used to relabel rewards.
            device: Device to store the data in
            n_envs: Number of parallel environments
            optimize_memory_usage: Enable a memory efficient variant
                of the replay buffer which reduces by almost a factor two the memory
                used, at a cost of more complexity.
            handle_timeout_termination: Handle timeout termination (due to timelimit)
                separately and treat the task as infinite horizon task.
                https://github.com/DLR-RM/stable-baselines3/issues/284
        """
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.reward_fn = reward_fn
        self.device = device

    def sample(self, *args, **kwargs):
        samples = self.sample(*args, **kwargs)

        rewards = self.reward_fn(**_samples_to_reward_fn_input(samples))

        shape = samples.rewards.shape
        rewards_th = util.safe_to_tensor(rewards).reshape(shape).to(self.device)

        return ReplayBufferSamples(
            samples.observations,
            samples.actions,
            samples.next_observations,
            samples.dones,
            rewards_th,
        )
