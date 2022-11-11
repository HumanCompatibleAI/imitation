"""Wrapper for reward labeling for transitions sampled from a replay buffer."""

from typing import Mapping, Type

import numpy as np
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
        *,
        replay_buffer_class: Type[ReplayBuffer],
        reward_fn: RewardFn,
        **kwargs,
    ):
        """Builds ReplayBufferRewardWrapper.

        Args:
            buffer_size: Max number of elements in the buffer
            observation_space: Observation space
            action_space: Action space
            replay_buffer_class: Class of the replay buffer.
            reward_fn: Reward function for reward relabeling.
            **kwargs: keyword arguments for ReplayBuffer.
        """
        # Note(yawen-d): we directly inherit ReplayBuffer and leave out the case of
        # DictReplayBuffer because the current RewardFn only takes in NumPy array-based
        # inputs, and SAC is the only use case for ReplayBuffer relabeling. See:
        # https://github.com/HumanCompatibleAI/imitation/pull/459#issuecomment-1201997194
        assert replay_buffer_class is ReplayBuffer, "only ReplayBuffer is supported"
        assert not isinstance(observation_space, spaces.Dict)
        self.replay_buffer = replay_buffer_class(
            buffer_size,
            observation_space,
            action_space,
            **kwargs,
        )
        self.reward_fn = reward_fn
        _base_kwargs = {k: v for k, v in kwargs.items() if k in ["device", "n_envs"]}
        super().__init__(buffer_size, observation_space, action_space, **_base_kwargs)

    @property
    def pos(self) -> int:
        return self.replay_buffer.pos

    @pos.setter
    def pos(self, pos: int):
        self.replay_buffer.pos = pos

    @property
    def full(self) -> bool:
        return self.replay_buffer.full

    @full.setter
    def full(self, full: bool):
        self.replay_buffer.full = full

    def sample(self, *args, **kwargs):
        samples = self.replay_buffer.sample(*args, **kwargs)
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

    def _get_samples(self):
        raise NotImplementedError(
            "_get_samples() is intentionally not implemented."
            "This method should not be called.",
        )


class ReplayBufferEntropyRewardWrapper(ReplayBuffer):
    """Relabel the rewards from a ReplayBuffer, initially using entropy as reward."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        *,
        replay_buffer_class: Type[ReplayBuffer],
        reward_fn: RewardFn,
        entropy_as_reward_samples: int,
        **kwargs,
    ):
        """Builds ReplayBufferRewardWrapper.

        Args:
            buffer_size: Max number of elements in the buffer
            observation_space: Observation space
            action_space: Action space
            replay_buffer_class: Class of the replay buffer.
            reward_fn: Reward function for reward relabeling.
            entropy_as_reward_samples: Number of samples to use entropy as the reward,
                before switching to using the reward_fn for relabeling.
            **kwargs: keyword arguments for ReplayBuffer.
        """
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            replay_buffer_class,
            reward_fn,
            **kwargs,
        )
        # TODO should we limit by number of batches (as this does)
        #      or number of observations returned?
        self.samples = 0
        self.entropy_as_reward_samples = entropy_as_reward_samples

    def sample(self, *args, **kwargs):
        self.samples += 1
        samples = super().sample(*args, **kwargs)
        if self.samples > self.entropy_as_reward_samples:
            return samples

        # TODO make the state entropy function accept batches
        # TODO compute state entropy for each reward
        # TODO replace the reward with the entropies
        # TODO note that we really ought to reset the reward network when we are done w/ entropy, and we have no business training it before then
