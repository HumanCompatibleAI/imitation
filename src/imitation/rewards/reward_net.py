"""Constructs deep network reward models."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

import gym
import numpy as np
import torch as th
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, preprocess_obs
from torch import nn

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

        if not (self.use_state or self.use_action or self.use_next_state):
            raise ValueError(
                "At least one of use_state, use_action, or use_next_state "
                "must be True"
            )

        self.base_reward_network = self.build_base_reward_network()

    @abstractmethod
    def _reward_train(
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

    def _reward_test(
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
        return self._reward_train(state, action, next_state, done)

    # FIXME(sam): rename this and the following method to predict_reward_train
    # and predict_reward_test, respectively. After that, remove underscores
    # from the names of the two methods above.
    def reward_train(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Compute the train reward with raw ndarrays, including preprocessing.

        Args:
          state: current state.
          action: action associated with `state`.
          next_state: next state.
          done: 0/1 value indicating whether episode terminates on transition
            to `next_state`.

        Returns:
          np.ndarray: A (None,) shaped ndarray holding
              the training reward associated with each timestep.
        """
        # FIXME(sam): reward_train/reward_test are ugly. Moving ndarrays to
        # devices and preprocessing them should not be the responsibility of
        # a reward network. Going to keep this for compat for now, but I want
        # to remove these methods in the future and push all the tensor/ndarray
        # conversion and preprocessing logic back up the call stack.
        return self._eval_reward(self._reward_train, state, action, next_state, done)

    def reward_test(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Compute the test reward with raw ndarrays, including preprocessing.

        Note this is the reward we use for transfer learning.

        Args:
          state: current state.
          action: action associated with `state`.
          next_state: next state.
          done: 0/1 value indicating whether episode terminates on transition
            to `next_state`.

        Returns:
          np.ndarray: A (None,) shaped ndarray holding the test reward
            associated with each timestep.
        """
        return self._eval_reward(self._reward_test, state, action, next_state, done)

    def _eval_reward(
        self,
        method: Callable[[th.Tensor] * 4, th.Tensor],
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Evaluates either train or test reward, given appropriate method."""
        # torchify
        dev = self.device()
        state_th = th.as_tensor(state, device=dev)
        action_th = th.as_tensor(action, device=dev)
        next_state_th = th.as_tensor(next_state, device=dev)
        done_th = th.as_tensor(done, device=dev)

        # preprocess observations/actions
        state_th = preprocess_obs(state_th, self.observation_space, self.scale)
        action_th = preprocess_obs(action_th, self.action_space, self.scale)
        next_state_th = preprocess_obs(
            next_state_th, self.observation_space, self.scale
        )
        done_th = done_th.to(th.float32)

        with th.no_grad():
            th_reward = self._reward_test(state_th, action_th, next_state_th, done_th)

        return th_reward.detach().cpu().numpy().squeeze(1)

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
        # FIXME(sam): this is ugly too. Remove it when/if I remove
        # reward_test()/reward_train().
        first_param = next(self.parameters())
        return first_param.device

    # TODO(sam): add these summaries back using
    # torch.util.tensorboard.SummaryWriter.
    # def build_summaries(self):
    #     tf.summary.histogram("train_reward", self._reward_train)
    #     tf.summary.histogram("test_reward", self._reward_test)


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

    def _reward_train(
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
        # TODO(sam): batch potential_network calls
        new_shaping_output = self.potential_network(next_state)
        old_shaping_output = self.potential_network(state)
        new_shaping = done * self._end_potential + (1 - done) * new_shaping_output
        return (
            base_reward_net_output
            + self._discount_factor * new_shaping
            - old_shaping_output
        )

    def _reward_test(
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

    # TODO(sam): add back these summary histograms too
    # def build_summaries(self):
    #     super().build_summaries()
    #     tf.summary.histogram("shaping_old", self._old_shaping_output)
    #     tf.summary.histogram("shaping_new", self._new_shaping_output)


class BasicRewardMLP(nn.Module):
    """MLP that flattens and concatenates current state, current action, next state, and
    done flag, depending on given `use_*` keyword arguments."""

    def __init__(
        self,
        observation_space,
        action_space,
        use_state,
        use_action,
        use_next_state,
        use_done,
        build_mlp_kwargs: Optional[Dict] = None,
    ):
        # XXX(sam): need docstring
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

        # XXX
        self.combined_size = combined_size
        self.observation_space = observation_space
        self.action_space = action_space

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

        return outputs


class BasicRewardNet(RewardNet):
    """An unshaped reward net with simple, default settings."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        base_reward_net_kwargs: Optional[dict] = None,
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

    def _reward_train(
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
        base_reward_net_kwargs: Optional[dict] = None,
        shaping_net_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Builds a simple shaped reward network.

        Args:
          observation_space: The observation space.
          action_space: The action space.
          base_reward_net_kwargs: Arguments passed to
            `build_mlp_base_reward_network`.
          shaping_net_kwargs: Arguments passed to `build_mlp_potential_network`.
          kwargs: Passed through to `RewardNetShaped`.
        """
        self.base_reward_net_kwargs = {
            "hid_sizes": (32, 32),
            **(base_reward_net_kwargs or {}),
        }
        self.shaping_net_kwargs = {
            "hid_sizes": (32, 32),
            **(base_reward_net_kwargs or {}),
        }
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
