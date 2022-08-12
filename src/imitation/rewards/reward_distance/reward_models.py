# =============================================================
# This file implements some basic reward models and also has
# some utils for making use of these reward models with
# imitation for epic.
# =============================================================
from __future__ import annotations

import abc
from collections.abc import MutableMapping
from typing import Dict, Optional

import gym
import torch
import numpy as np

class RewardModel(abc.ABC):
    def reward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: Optional[torch.Tensor],
            terminals: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Computes the reward for the associated transitions.

        We assume that all reward models operate on `torch.Tensor`s.

        Args:
            states: The states.
            actions: The actions.
            next_states: The next states. Some reward models don't use these so they're optional.
            terminals: Indicators for whether the transition ended an episode.
                Some reward models don't use these so they're optional.

        Returns:
            Tensor of scalar reward values.
        """


class ConstantRewardModel(RewardModel):
    """A reward model that returns a constant reward for all 
       (s,a,s') pairs.

    """
    def __init__(self, constant_reward: float):
        super().__init__()
        self.constant_reward = constant_reward

    def reward(self,
               states: torch.Tensor,
               actions: torch.Tensor,
               next_states: Optional[torch.Tensor],
               terminals: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """See base class documentation.

        """
        assert states.shape[0] == actions.shape[0] == next_states.shape[0]
        if terminals is not None:
            assert states.shape[0] == terminals.shape[0]
        batch_size = states.shape[0]
        return torch.ones((batch_size))*self.constant_reward



class ZeroRewardModel(ConstantRewardModel):
    """A reward model that returns a ZERO reward for all 
       (s,a,s') pairs.

    """
    def __init__(self, noisy_reward=False):
        super().__init__(0.0)
        self.noisy_reward = noisy_reward

    def reward(self,
               states: torch.Tensor,
               actions: torch.Tensor,
               next_states: Optional[torch.Tensor],
               terminals: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """See base class documentation.

        """
        orig_reward = super().reward(states, actions, next_states, terminals)
        if self.noisy_reward:
            return orig_reward + torch.rand_like(orig_reward)
        else:
            return orig_reward



# TODO: Compare outputs with HC environment to make sure that the implementation
# of reward function here is correct
class GroundTruthRewardModelForHC(RewardModel):
    """
    Implements ground truth reward for half cheetah environment.
    Alternative to doing my own implementation here is to make calls
    to mujoco environment. There are few issues with that
    (1) Mujoco environments do not have explicit API for getting reward
    (2) It is likely to be much slower due to inability to make
        batched calls to mujoco environment without having the overhead
        of multiprocessing.

    Args:
        make_env_fn: Callable which returns the desired gym environment.
    """
    def __init__(self, make_env_fn):
        env = make_env_fn()
        self.dt = env.unwrapped.dt
        self.states_space_dim = env.observation_space.shape[0]
        self.action_space_dim = env.action_space.shape[0]
        del(env)

    def reward(self,
               states: torch.Tensor,
               actions: torch.Tensor,
               next_states: Optional[torch.Tensor],
               terminals: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """See base class documentation.

        """
        assert states.shape[0] == actions.shape[0] == next_states.shape[0]
        if terminals is not None:
            assert states.shape[0] == terminals.shape[0]
        assert states.shape[1] == self.states_space_dim
        assert next_states.shape[1] == self.states_space_dim
        assert actions.shape[1] == self.action_space_dim

        xpos_before = states[:,0]
        xpos_after  = next_states[:,0]
        reward_run  = (xpos_after - xpos_before)/self.dt
        reward_ctrl = -0.1 * actions.square().sum(axis=1)
        reward = reward_run + reward_ctrl
        return reward



class RandomRewardModel(RewardModel):
    """
    Returns a random value for each input.

    """
    def __init__(self):
        pass

    def reward(self,
               states: torch.Tensor,
               actions: torch.Tensor,
               next_states: Optional[torch.Tensor],
               terminals: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """See base class documentation.

        """
        return torch.randn((len(states)))


class RewardModelWrapperForImitation(RewardModel):
    """Creates a wrapper around reward_fn provided by imitation
       to maintain a consistent api for epic.

    """
    def __init__(self, reward_fn):
        self.reward_fn = reward_fn

    def reward(self,
               states: torch.Tensor,
               actions: torch.Tensor,
               next_states: Optional[torch.Tensor],
               terminals: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """See base class documentation.

        """
        if terminals is None:
            terminals = torch.zeros_like(states)
        assert len(states) == len(actions) == len(next_states) == len(terminals)
        return torch.from_numpy(self.reward_fn(states.numpy()[:,1:],
                                actions.numpy(),
                                next_states.numpy()[:,1:],
                                terminals.numpy()))



