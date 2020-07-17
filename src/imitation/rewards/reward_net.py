"""Constructs deep network reward models."""

import collections
from abc import ABC, abstractmethod
from typing import Mapping, Optional

import gym
import numpy as np
import torch as th
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from torch import nn

import imitation.rewards.common as rewards_common
from imitation.util import networks


class RewardNet(nn.Module, ABC):
    """Abstract reward network.

    Attributes:
      observation_space: The observation space.
      action_space: The action space.
      base_reward_network (nn.Module): neural network taking state, action, next
        state and dones, and producing a reward value.
      use_state: should `base_reward_network` pay attention to current state?
      use_next_state: should `base_reward_network` pay attention to next state?
      use_action: should `base_reward_network` pay attention to action?
      use_done: should `base_reward_network` pay attention to done flags?
      scale: should inputs be scaled to lie in [0,1] using space bounds?
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        scale: bool = False,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
    ):
        """Builds a reward network.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            scale: Whether to scale the input.
            use_state: Whether state is included in inputs to network.
            use_action: Whether action is included in inputs to network.
            use_next_state: Whether next state is included in inputs to network.
            use_done: Whether episode termination is included in inputs to network.
        """
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.scale = scale
        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        self.use_done = use_done

        if not (
            self.use_state or self.use_action or self.use_next_state or self.use_done
        ):
            raise ValueError(
                "At least one of use_state, use_action, use_next_state or use_done "
                "must be True"
            )

        self.base_reward_network = self.build_base_reward_network()

    @abstractmethod
    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """A Tensor holding the training reward associated with each timestep.

        This performs inner logic for `self.reward_train()`. See
        `reward_train()` docs for explanation of arguments and return values.
        """

    def reward_test(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """A Tensor holding the test reward associated with each timestep.

        This performs inner logic for `self.reward_test()`. See
        `reward_test()` docs for explanation of arguments and return values.
        """
        return self.reward_train(state, action, next_state, done)

    def predict_reward_train(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Compute the train reward with raw ndarrays, including preprocessing.

        Args:
          state: current state. Leading dimension should be batch size B.
          action: action associated with `state`.
          next_state: next state.
          done: 0/1 value indicating whether episode terminates on transition
            to `next_state`.

        Returns:
          np.ndarray: A (B,)-shaped ndarray holding
              the training reward associated with each timestep.
        """
        return self._eval_reward(True, state, action, next_state, done)

    def predict_reward_test(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Compute the test reward with raw ndarrays, including preprocessing.

        Note this is the reward we use for transfer learning.

        Args:
          state: current state. Lead dimension should be batch size B.
          action: action associated with `state`.
          next_state: next state.
          done: 0/1 value indicating whether episode terminates on transition
            to `next_state`.

        Returns:
          np.ndarray: A (B,)-shaped ndarray holding the test reward
            associated with each timestep.
        """
        return self._eval_reward(False, state, action, next_state, done)

    def _eval_reward(
        self,
        is_train: bool,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Evaluates either train or test reward, given appropriate method."""
        (
            state_th,
            action_th,
            next_state_th,
            done_th,
        ) = rewards_common.disc_rew_preprocess_inputs(
            observation_space=self.observation_space,
            action_space=self.action_space,
            state=state,
            action=action,
            next_state=next_state,
            done=done,
            device=self.device(),
            scale=self.scale,
        )

        with th.no_grad():
            if is_train:
                rew_th = self.reward_train(state_th, action_th, next_state_th, done_th)
            else:
                rew_th = self.reward_test(state_th, action_th, next_state_th, done_th)

        rew = rew_th.detach().cpu().numpy().flatten()
        assert rew.shape == state.shape[:1]
        return rew

    @abstractmethod
    def build_base_reward_network(self) -> nn.Module:
        """Builds the test reward network.

        The output of the network is treated as a reward suitable for transfer
        learning. The network will be provided with the current observation,
        current action, next observation, and done flag, as indicated by
        self.{use_state, use_action, use_next_state, use_done}.

        Returns: an `nn.Module` which takes the appropriate inputs (described
            above) and returns a scalar reward for each batch element.
        """

    def device(self) -> th.device:
        """Use a heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device


class RewardNetShaped(RewardNet):
    """Abstract reward network with a phi network to shape training reward.

    This RewardNet formulation matches Equation (4) in the AIRL paper.
    Note that the experiments in Table 2 of the same paper showed shaped
    training rewards to be inferior to an unshaped training rewards in
    a Pendulum environment imitation learning task (and maybe HalfCheetah).
    (See original implementation of Pendulum experiment's reward function at
    https://github.com/justinjfu/inverse_rl/blob/master/inverse_rl/models/imitation_learning.py#L374)

    To make a concrete subclass, implement `build_potential_network()` and
    `build_base_reward_network()`.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        discount_factor: float = 0.99,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, **kwargs)
        self._discount_factor = discount_factor

        self.potential_network = self.build_potential_network()
        # end_potential is the potential when the episode terminates.
        if discount_factor == 1.0:
            # If undiscounted, terminal state must have potential 0.
            self.end_potential = 0.0
        else:
            # Otherwise, it can be arbitrary, so make a trainable variable.
            self.end_potential = nn.Parameter(th.zeros(()))

    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Compute the (shaped) training reward of each timestep."""
        base_reward_net_output = self.base_reward_network(
            state, action, next_state, done
        )
        new_shaping_output = self.potential_network(next_state).flatten()
        old_shaping_output = self.potential_network(state).flatten()
        done_f = done.float()
        new_shaping = done_f * old_shaping_output + (1 - done_f) * new_shaping_output
        final_rew = (
            base_reward_net_output
            + self._discount_factor * new_shaping
            - old_shaping_output
        )
        assert final_rew.shape == state.shape[:1]
        return final_rew

    def reward_test(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Compute the (unshaped) test reward associated with each timestep."""
        return self.base_reward_network(state, action, next_state, done)

    @abstractmethod
    def build_potential_network(self) -> nn.Module:
        """Build the reward shaping network (disentangles dynamics from reward).

        Returns:
          An `nn.Module` mapping from observations to potential values.
        """


class BasicRewardMLP(nn.Module):
    """MLP that flattens and concatenates current state, current action, next state, and
    done flag, depending on given `use_*` keyword arguments."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool,
        use_action: bool,
        use_next_state: bool,
        use_done: bool,
        build_mlp_kwargs: Optional[Mapping] = None,
    ):
        """Builds reward MLP.

        Args:
          observation_space: The observation space.
          action_space: The action space.
          use_state: should the current state be included as an input to the MLP?
          use_action: should the current action be included as an input to the MLP?
          use_next_state: should the next state be included as an input to the MLP?
          use_done: should the "done" flag be included as an input to the MLP?
          build_mlp_kwargs: Arguments passed to `build_mlp`.
        """
        super().__init__()
        combined_size = 0

        self.use_state = use_state
        if self.use_state:
            combined_size += get_flattened_obs_dim(observation_space)

        self.use_action = use_action
        if self.use_action:
            combined_size += get_flattened_obs_dim(action_space)

        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size += get_flattened_obs_dim(observation_space)

        self.use_done = use_done
        if self.use_done:
            combined_size += 1

        full_build_mlp_kwargs = {
            "in_size": combined_size,
            "hid_sizes": (32, 32),
        }
        full_build_mlp_kwargs.update(build_mlp_kwargs)

        self.mlp = networks.build_mlp(**full_build_mlp_kwargs)

    def forward(self, state, action, next_state, done):
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(state, 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(next_state, 1))
        if self.use_done:
            inputs.append(th.flatten(done, 1))

        inputs_concat = th.cat(inputs, dim=1)

        outputs = self.mlp(inputs_concat)
        flat_outputs = outputs.squeeze(1)
        assert flat_outputs.shape == state.shape[:1]

        return flat_outputs


class BasicRewardNet(RewardNet):
    """An unshaped reward net with simple, default settings."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        base_reward_net_kwargs: Optional[Mapping] = None,
        **kwargs,
    ):
        """Builds a simple reward network.

        Args:
          observation_space: The observation space.
          action_space: The action space.
          base_reward_net_kwargs: Arguments passed to `build_mlp_base_reward_network`.
          kwargs: Passed through to RewardNet.
        """
        self.base_reward_net_kwargs = {
            "hid_sizes": (32, 32),
            **(base_reward_net_kwargs or {}),
        }
        RewardNet.__init__(self, observation_space, action_space, **kwargs)

    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Compute the train reward associated with each timestep."""
        return self.base_reward_network(state, action, next_state, done)

    def build_base_reward_network(self) -> nn.Module:
        return BasicRewardMLP(
            observation_space=self.observation_space,
            action_space=self.action_space,
            use_state=self.use_state,
            use_action=self.use_action,
            use_next_state=self.use_next_state,
            use_done=self.use_done,
            build_mlp_kwargs=self.base_reward_net_kwargs,
        )


class BasicShapedRewardNet(RewardNetShaped):
    """A shaped reward network with simple, default settings.

    With default parameters this RewardNet has two hidden layers [32, 32]
    for the base reward network and shaping network.

    This network is feed-forward and flattens inputs, so is a poor choice for
    training on pixel observations.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        base_reward_net_kwargs: Optional[Mapping] = None,
        shaping_net_kwargs: Optional[Mapping] = None,
        build_mlp_kwargs: Optional[Mapping] = None,
        **kwargs,
    ):
        """Builds a simple shaped reward network.

        Args:
          observation_space: The observation space.
          action_space: The action space.
          base_reward_net_kwargs: Arguments passed to
            `build_mlp_base_reward_network`.
          shaping_net_kwargs: Arguments passed to `build_mlp_potential_network`.
          build_mlp_kwargs: Arguments passed to both base reward and potential
            network constructors. Values in this dict will be overridden by
            base_reward_net_kwargs and shaping_net_kwargs.
          kwargs: Passed through to `RewardNetShaped`.
        """
        self.base_reward_net_kwargs = collections.ChainMap(
            base_reward_net_kwargs or {},
            build_mlp_kwargs or {},
            {"hid_sizes": (32, 32)},
        )
        self.shaping_net_kwargs = collections.ChainMap(
            shaping_net_kwargs or {}, build_mlp_kwargs or {}, {"hid_sizes": (32, 32)}
        )
        RewardNetShaped.__init__(
            self, observation_space, action_space, **kwargs,
        )

    def build_base_reward_network(self):
        return BasicRewardMLP(
            observation_space=self.observation_space,
            action_space=self.action_space,
            use_state=self.use_state,
            use_action=self.use_action,
            use_next_state=self.use_next_state,
            use_done=self.use_done,
            build_mlp_kwargs=self.base_reward_net_kwargs,
        )

    def build_potential_network(self):
        in_size = get_flattened_obs_dim(self.observation_space)
        return networks.build_mlp(in_size=in_size, **self.shaping_net_kwargs)
