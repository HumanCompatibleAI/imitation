"""Wrapper for reward labeling for transitions sampled from a replay buffer."""

from typing import Callable, Mapping, Type

import numpy as np
from gym import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from imitation.rewards.reward_function import RewardFn, ReplayBufferAwareRewardFn
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


class ReplayBufferView:
    """A read-only view over a valid records in a ReplayBuffer.

    Args:
        observations_buffer: Array buffer holding observations
        buffer_slice_provider: Function returning slice of buffer
            with valid observations
    """

    def __init__(
        self,
        observations_buffer: np.ndarray,
        buffer_slice_provider: Callable[[], slice],
    ):
        self._observations_buffer_view = observations_buffer.view()
        self._observations_buffer_view.flags.writeable = False
        self._buffer_slice_provider = buffer_slice_provider

    @property
    def observations(self):
        return self._observations_buffer_view[self._buffer_slice_provider()]


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
            on_initialized_callback: Callback called with reference to this object after
                this instance is fully initialized. This provides a hook to access the
                buffer after it is created from inside a Stable Baselines algorithm.
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
        if isinstance(reward_fn, ReplayBufferAwareRewardFn):
            reward_fn.on_replay_buffer_initialized(self)

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

    @property
    def buffer_view(self) -> ReplayBufferView:
        def valid_buffer_slice():
            return slice(None) if self.full else slice(self.pos)

        return ReplayBufferView(self.replay_buffer.observations, valid_buffer_slice)

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
