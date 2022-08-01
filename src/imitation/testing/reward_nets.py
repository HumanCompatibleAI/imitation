"""Utility functions for testing reward nets."""

import gym

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
