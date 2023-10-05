"""Wrapper for reward labeling for transitions sampled from a replay buffer."""

from typing import Mapping, Type

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
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


def _rollout_buffer_to_reward_fn_input(
    buffer: RolloutBuffer,
) -> Mapping[str, np.ndarray]:
    """Convert a sample from a rollout buffer to a numpy array."""
    assert buffer.observations is not None
    assert buffer.actions is not None
    obs = buffer.observations
    next_obs = obs[1:]
    next_obs = np.concatenate([next_obs, obs[-1:]], axis=0)  # last obs not available
    actions = buffer.actions
    dones = buffer.episode_starts
    dones = np.roll(dones, -1, axis=0)
    dones[-1] = np.ones_like(dones[-1])  # last dones not available

    return dict(
        state=obs.reshape(-1, *obs.shape[2:]),
        action=actions.reshape(-1, *actions.shape[2:]),
        next_state=next_obs.reshape(-1, *next_obs.shape[2:]),
        done=dones.reshape(-1),
    )


def _replay_buffer_to_reward_fn_input(
    buffer: ReplayBuffer,
) -> Mapping[str, np.ndarray]:
    """Convert a sample from a replay buffer to a numpy array."""
    assert buffer.observations is not None
    assert buffer.next_observations is not None
    assert buffer.actions is not None
    obs = buffer.observations
    next_obs = buffer.next_observations
    actions = buffer.actions
    dones = buffer.dones

    return dict(
        state=obs.reshape(-1, *obs.shape[2:]),
        action=actions.reshape(-1, *actions.shape[2:]),
        next_state=next_obs.reshape(-1, *next_obs.shape[2:]),
        done=dones.reshape(-1),
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
