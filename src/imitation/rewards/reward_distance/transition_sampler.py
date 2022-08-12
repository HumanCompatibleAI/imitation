import logging
import math
from multiprocessing import cpu_count, Pipe, Process
from multiprocessing.connection import Connection
from typing import Callable, Tuple, Optional

import abc
import itertools

import gym
import numpy as np
import torch

class TransitionSampler(abc.ABC):
    """Base class for objects that sample transitions (actions and next states) from provided states.

    Design note: I considered breaking this class into two classes (a policy class and transition function
    class); however, there are cases where, for the purposes of EPIC, you might want to sample the actions
    and next states jointly, and in that case having two separate classes complicates things significantly.
    """
    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples transitions for the provided states.

        A "transition" includes an action, resulting next state, and (potentially) a weighting on the triple
        indicating how likely it is (though the weight might not be a normalized probability).

        Args:
            states: The states from which to sample transitions. Shape of (batch_size, ...), where ... indicates
                some arbitrary state dimensions.

        Returns:
            A tuple of tensors. The first being the actions of shape (batch_size, num_transitions_per_state, ...),
            the second being the next states of shape (batch_size, num_transitions_per_state, ...), and the third
            being the weights associated with the overall triple of shape (batch_size, num_transitions_per_state).
        """
    @property
    @abc.abstractmethod
    def num_transitions_per_state(self) -> int:
        """Returns the number of transitions sampled per state."""


def add_dim_and_tile_first_dim(arr: torch.Tensor, reps: int) -> torch.Tensor:
    """Adds a dimension to `arr` and then tiles it `reps` times.

    For example, if array is of shape (10, 20) and reps is 5, this will output a tensor of shape (5, 10, 20).

    Args:
        arr: Tensor to tile.
        reps: The number of repetitions of the array.

    Retruns:
        Tensor that's the tiled version of `arr`, but with a dimension added.
    """
    # Note the use of `arr.ndim` instead of `arr.ndim - 1` because a dimension will be added.
    dim_reps = (reps, ) + (1, ) * arr.ndim
    return torch.tile(arr[None], dim_reps)


class FixedDistributionTransitionSampler(TransitionSampler):
    """A transition sampler that returns a fixed / constant distribution independent of the provided states.

    Args:
        actions: The actions to provide in response to all states.
            Shape should be (num_transitions_per_state, ...) where "..." indicates arbitrary number of action dimensions.
        next_states: The next states to provide in response to all states.
            Shape should be (num_transitions_per_state, ...) where "..." indicates arbitrary number of state dimensions.
        weights: The weights to associate with those actions and next states. If `None` a tensor of ones is used.
            Shape should be (num_transitions_per_state,) if provided.
    """
    def __init__(self, actions: torch.Tensor, next_states: torch.Tensor, weights: Optional[torch.Tensor] = None):
        assert len(actions) == len(next_states)
        self.actions = actions
        self.next_states = next_states
        if weights is None:
            weights = torch.ones(len(next_states), dtype=next_states.dtype, device=actions.device)
        self.weights = weights

        # Cache the tiled versions of these.
        self.tiled_actions = None
        self.tiled_next_states = None
        self.tiled_weights = None

    @property
    def num_transitions_per_state(self) -> int:
        return len(self.actions)

    def sample(
            self,
            states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tiles the fixed distribution of actions and next states to match the size of the provided states.

        See base class documentation for details on the args and return values.

        Design note: This class caches the tiled actions / next states / weights because they are likely to
        be used repeatedly in batched computations.
        """
        if not self._tiling_is_initialized_and_matches_states(states):
            num_states = len(states)
            self.tiled_actions = add_dim_and_tile_first_dim(self.actions, num_states)
            self.tiled_next_states = add_dim_and_tile_first_dim(self.next_states, num_states)
            self.tiled_weights = add_dim_and_tile_first_dim(self.weights, num_states)
            assert self._tiling_is_initialized_and_matches_states(states)

        return self.tiled_actions, self.tiled_next_states, self.tiled_weights

    def _tiling_is_initialized_and_matches_states(self, states: torch.Tensor) -> bool:
        """Returns True if the tiling (of actions, next states, weights) is initialized and matches the states size.

        Args:
            states: The states to match.
        """
        if self.tiled_actions is None or self.tiled_next_states is None or self.tiled_weights is None:
            return False

        assert len(self.tiled_actions) == len(self.tiled_next_states)
        assert len(self.tiled_next_states) == len(self.tiled_weights)
        if len(states) != len(self.tiled_actions):
            return False

        return True


def repeat_interleave_and_reshape_to_3d(arr: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat-interleaves a tensor and then reshapes it to add a dimension for the repeats.

    Args:
        arr: 2D tensor, 0th dim of which should be repeat-interleaved.
        repeats: The number of times to repeat the tensor.

    Returns:
        Tensor with 0th dim repeat-interleaved. So if arr has shape (5, 10) and repeats is 3, this
        will return a tensor of shape (5, 3, 10).
    """
    assert arr.ndim == 2
    return torch.repeat_interleave(arr, repeats, dim=0).reshape(arr.shape[0], repeats, arr.shape[1])


# ====================================================================
# How to sample actions?
# uniformly ----------------> UniformlyRandomActionSampler
# maximum
# ====================================================================


class ActionSampler(abc.ABC):
    """Base class for classes that sample actions.

    This class is used internally in the implementation of some transition samplers (in particular
    those that break up transition sampling into sampling actions and then subsequently next states).

    It's conceptually different from a policy because we sample many actions jointly.
    """
    @property
    @abc.abstractmethod
    def num_actions(self) -> int:
        """Returns the number of actions that will be sampled."""
    @abc.abstractmethod
    def __call__(self, size: int, dtype: type, device: torch.device) -> torch.Tensor:
        """Samples actions for some number (`size` states)."""


class UniformlyRandomActionSampler(ActionSampler):
    """Samples actions uniformly at random from a possible range.

    Args:
        num_actions: The number of actions to sample.
        max_magnitude: The max magnitude along each axis.
        action_dim: The dim of the actions.
    """
    def __init__(self, num_actions: int, max_magnitude: float, action_dim: int = 2):
        self._num_actions = num_actions
        self.max_magnitude = max_magnitude
        self.action_dim = action_dim

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def __call__(self, size: int, dtype: type, device: torch.device) -> torch.Tensor:
        rand = torch.rand((size, self.num_actions, self.action_dim), dtype=dtype, device=device)
        return -self.max_magnitude + rand * self.max_magnitude * 2


class BoundaryActionSampler(ActionSampler):
    """Samples actions from the boundary of the possible range along each axis.

    Args:
        max_magnitude: The max magnitude along each axis.
    """
    def __init__(self, max_magnitude: float):
        # The extra dimension is to make tiling simpler.
        self._actions = torch.tensor([[
            [max_magnitude, max_magnitude],
            [-max_magnitude, max_magnitude],
            [max_magnitude, -max_magnitude],
            [-max_magnitude, -max_magnitude],
        ]])
        self._num_actions = self._actions.shape[1]

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def __call__(self, size: int, dtype: type, device: torch.device) -> torch.Tensor:
        return torch.tile(self._actions, (size, 1, 1)).to(dtype).to(device)


class LinearActionSampler(ActionSampler):
    """Samples actions in discretized increments linearly between boundaries.

    Args:
        num_actions_each_dim: The number of actions to sample along each dimension.
        max_magnitude: The max magnitude along each axis.
        ndim: Number of dimensions in the action space.
    """
    def __init__(self, num_actions_each_dim: int, max_magnitude: float, ndim: int = 2):
        assert num_actions_each_dim > 1

        self.ndim = ndim
        dim_actions = [np.linspace(-max_magnitude, max_magnitude, num_actions_each_dim) for _ in range(ndim)]
        self._actions = torch.tensor(list(itertools.product(*dim_actions)))
        # The extra dimension is to make tiling simpler.
        self._actions = self._actions[None]
        self._num_actions = self._actions.shape[1]

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def __call__(self, size: int, dtype: type, device: torch.device) -> torch.Tensor:
        return torch.tile(self._actions, (size, 1, 1)).to(dtype).to(device)