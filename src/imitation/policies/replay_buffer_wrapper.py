"""Wrapper for reward labeling for transitions sampled from a replay buffer."""

from typing import Mapping, Type

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer, RolloutBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from imitation.rewards.reward_function import RewardFn
from imitation.util import util


def _replay_samples_to_reward_fn_input(
    samples: ReplayBufferSamples,
) -> Mapping[str, np.ndarray]:
    """Convert a sample from a replay buffer to a numpy array."""
    return dict(
        state=samples.observations.cpu().numpy(),
        action=samples.actions.cpu().numpy(),
        next_state=samples.next_observations.cpu().numpy(),
        done=samples.dones.cpu().numpy(),
    )


def _rollout_samples_to_reward_fn_input(
    buffer: RolloutBuffer,
) -> Mapping[str, np.ndarray]:
    """Convert a sample from a rollout buffer to a numpy array."""
    return dict(
        state=buffer.observations,
        action=buffer.actions,
        next_state=buffer.next_observations,
        done=buffer.dones,
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
        assert (
            replay_buffer_class is ReplayBuffer
        ), f"only ReplayBuffer is supported: given {replay_buffer_class}"
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
        rewards = self.reward_fn(**_replay_samples_to_reward_fn_input(samples))
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


class RolloutBufferRewardWrapper(BaseBuffer):
    """Relabel the rewards in transitions sampled from a RolloutBuffer."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        *,
        rollout_buffer_class: Type[RolloutBuffer],
        reward_fn: RewardFn,
        **kwargs,
    ):
        """Builds RolloutBufferRewardWrapper.

        Args:
            buffer_size: Max number of elements in the buffer
            observation_space: Observation space
            action_space: Action space
            rollout_buffer_class: Class of the rollout buffer.
            reward_fn: Reward function for reward relabeling.
            **kwargs: keyword arguments for RolloutBuffer.
        """
        # Note(yawen-d): we directly inherit RolloutBuffer and leave out the case of
        # DictRolloutBuffer because the current RewardFn only takes in NumPy array-based
        # inputs, and GAIL/AIRL is the only use case for RolloutBuffer relabeling. See:
        # https://github.com/HumanCompatibleAI/imitation/pull/459#issuecomment-1201997194
        assert rollout_buffer_class is RolloutBuffer, "only RolloutBuffer is supported"
        assert not isinstance(observation_space, spaces.Dict)
        self.rollout_buffer = rollout_buffer_class(
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
        return self.rollout_buffer.pos

    @property
    def values(self):
        return self.rollout_buffer.values

    @property
    def observations(self):
        return self.rollout_buffer.observations

    @property
    def actions(self):
        return self.rollout_buffer.actions

    @property
    def log_probs(self):
        return self.rollout_buffer.log_probs

    @property
    def advantages(self):
        return self.rollout_buffer.advantages

    @property
    def rewards(self):
        return self.rollout_buffer.rewards

    @property
    def returns(self):
        return self.rollout_buffer.returns

    @pos.setter
    def pos(self, pos: int):
        self.rollout_buffer.pos = pos

    @property
    def full(self) -> bool:
        return self.rollout_buffer.full

    @full.setter
    def full(self, full: bool):
        self.rollout_buffer.full = full

    def reset(self):
        self.rollout_buffer.reset()

    def get(self, *args, **kwargs):
        if not self.rollout_buffer.generator_ready:
            input_dict = _rollout_samples_to_reward_fn_input(self.rollout_buffer)
            rewards = np.zeros_like(self.rollout_buffer.rewards)
            for i in range(self.buffer_size):
                rewards[i] = self.reward_fn(**{k: v[i] for k, v in input_dict.items()})

            self.rollout_buffer.rewards = rewards
            self.rollout_buffer.compute_returns_and_advantage(
                self.last_values, self.last_dones
            )
        ret = self.rollout_buffer.get(*args, **kwargs)
        return ret

    def add(self, *args, **kwargs):
        self.rollout_buffer.add(*args, **kwargs)

    def _get_samples(self):
        raise NotImplementedError(
            "_get_samples() is intentionally not implemented."
            "This method should not be called.",
        )

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: np.ndarray
    ) -> None:
        self.last_values = last_values
        self.last_dones = dones
        self.rollout_buffer.compute_returns_and_advantage(last_values, dones)
