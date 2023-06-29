"""Utility functions for testing reward nets."""

import gymnasium as gym
import torch as th

from imitation.rewards import reward_nets


def make_ensemble(
    obs_space: gym.Env,
    action_space: gym.Env,
    num_members: int = 2,
    **kwargs,
):
    """Create a simple reward ensemble."""
    return reward_nets.RewardEnsemble(
        obs_space,
        action_space,
        members=[
            reward_nets.BasicRewardNet(obs_space, action_space, **kwargs)
            for _ in range(num_members)
        ],
    )


class MockRewardNet(reward_nets.RewardNet):
    """A mock reward net for testing."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        value: float = 0.0,
    ):
        """Create mock reward.

        Args:
            observation_space: observation space of the env
            action_space: action space of the env
            value: The reward to always return. Defaults to 0.0.
        """
        super().__init__(observation_space, action_space)
        self.value = value

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        batch_size = state.shape[0]
        return th.full(
            (batch_size,),
            fill_value=self.value,
            dtype=th.float32,
            device=state.device,
        )
