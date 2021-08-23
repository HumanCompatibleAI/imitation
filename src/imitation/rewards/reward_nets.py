"""Constructs deep network reward models."""

import abc
from typing import Callable, Iterable, Sequence

import gym
import numpy as np
import torch as th
from stable_baselines3.common import preprocessing
from torch import nn

import imitation.rewards.common as rewards_common
from imitation.util import networks


class RewardNet(nn.Module, abc.ABC):
    """Minimal abstract reward network.

    Only requires the implementation of a forward pass (calculating rewards given
    a batch of states, actions, next states and dones).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        normalize_images: bool = True,
    ):
        """Initialize the RewardNet.

        Args:
            observation_space (gym.Space): the observation space of the environment
            action_space (gym.Space): the action space of the environment
            normalize_images (bool, optional): whether to automatically normalize
                image observations to [0, 1] (from 0 to 255). Defaults to True.
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.normalize_images = normalize_images

    @abc.abstractmethod
    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ):
        """Compute rewards for a batch of transitions and keep gradients."""

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Compute rewards for a batch of transitions without gradients.
        Also performs some preprocessing and numpy conversion.
        """
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
            device=self.device,
            scale=self.normalize_images,
        )

        with th.no_grad():
            rew_th = self(state_th, action_th, next_state_th, done_th)

        rew = rew_th.detach().cpu().numpy().flatten()
        assert rew.shape == state.shape[:1]
        return rew

    @property
    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        try:
            first_param = next(self.parameters())
            return first_param.device
        except StopIteration:
            # if the model has no parameters, we use the CPU
            return th.device("cpu")


class ShapedRewardNet(RewardNet):
    """A RewardNet consisting of a base net and a potential shaping."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        base: RewardNet,
        potential: Callable[[th.Tensor], th.Tensor],
        discount_factor: float,
        normalize_images: bool = True,
    ):
        """Setup a ShapedRewardNet instance.

        Args:
            observation_space (gym.Space): observation space of the environment
            action_space (gym.Space): action space of the environment
            base (RewardNet): the base reward net to which the potential shaping
                will be added.
            potential (Callable[[th.Tensor], th.Tensor]): A callable which takes
                a batch of states (as a PyTorch tensor) and returns a batch of
                potentials for these states. If this is a PyTorch Module, it becomes
                a submodule of the ShapedRewardNet instance.
            discount_factor (float): discount factor to use for the potential shaping
            normalize_images (bool, optional): passed through to `RewardNet.__init__`,
                see its documentation
        """
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            normalize_images=normalize_images,
        )
        self.base = base
        self.potential = potential
        self.discount_factor = discount_factor

        # end_potential is the potential when the episode terminates.
        if discount_factor == 1.0:
            # If undiscounted, terminal state must have potential 0.
            self.end_potential = 0.0
        else:
            # Otherwise, it can be arbitrary, so make a trainable variable.
            self.end_potential = nn.Parameter(th.zeros(()))

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ):
        base_reward_net_output = self.base(state, action, next_state, done)
        new_shaping_output = self.potential(next_state).flatten()
        old_shaping_output = self.potential(state).flatten()
        done_f = done.float()
        new_shaping = done_f * self.end_potential + (1 - done_f) * new_shaping_output
        final_rew = (
            base_reward_net_output
            + self.discount_factor * new_shaping
            - old_shaping_output
        )
        assert final_rew.shape == state.shape[:1]
        return final_rew


class AIRLRewardNet(nn.Module):
    """Wrapper around RewardNet with different forward passes for test and train.

    This class is used for AIRL, where we want the potential shaping to be included
    during training but not testing. If the wrapped RewardNet is not an instance
    of ShapedRewardNet, then the test and train passes will be identical.

    TODO(ejnnr): get rid of this class entirely, instead just implement the logic
        inside AIRL itself?
    """

    def __init__(self, reward_net: RewardNet):
        """Builds a reward network for AIRL.

        Args:
            reward_net: the RewardNet instance to be wrapped
        """
        super().__init__()
        self.reward_net = reward_net
        self.observation_space = reward_net.observation_space
        self.action_space = reward_net.action_space
        if isinstance(reward_net, ShapedRewardNet):
            self.test_reward_net = reward_net.base
        else:
            self.test_reward_net = reward_net

    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """A Tensor holding the training reward associated with each timestep.

        This performs inner logic for `self.predict_reward_train()`. See
        `predict_reward_train()` docs for explanation of arguments and return values.
        """
        return self.reward_net(state, action, next_state, done)

    def reward_test(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """A Tensor holding the test reward associated with each timestep.

        This performs inner logic for `self.predict_reward_test()`. See
        `predict_reward_test()` docs for explanation of arguments and return
        values.
        """
        return self.test_reward_net(state, action, next_state, done)

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
        return self.reward_net.predict(state, action, next_state, done)

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
        return self.test_reward_net.predict(state, action, next_state, done)


class BasicRewardNet(RewardNet):
    """MLP that flattens and concatenates current state, current action, next state, and
    done flag, depending on given `use_*` keyword arguments."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        **kwargs,
    ):
        """Builds reward MLP.

        Args:
          observation_space: The observation space.
          action_space: The action space.
          use_state: should the current state be included as an input to the MLP?
          use_action: should the current action be included as an input to the MLP?
          use_next_state: should the next state be included as an input to the MLP?
          use_done: should the "done" flag be included as an input to the MLP?
          kwargs: passed straight through to build_mlp.
        """
        super().__init__(observation_space, action_space)
        combined_size = 0

        self.use_state = use_state
        if self.use_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_action = use_action
        if self.use_action:
            combined_size += preprocessing.get_flattened_obs_dim(action_space)

        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_done = use_done
        if self.use_done:
            combined_size += 1

        full_build_mlp_kwargs = {
            "hid_sizes": (32, 32),
        }
        full_build_mlp_kwargs.update(kwargs)
        full_build_mlp_kwargs.update(
            {
                # we do not want these overridden
                "in_size": combined_size,
                "out_size": 1,
                "squeeze_output": True,
            }
        )

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
            inputs.append(th.reshape(done, [-1, 1]))

        inputs_concat = th.cat(inputs, dim=1)

        outputs = self.mlp(inputs_concat)
        assert outputs.shape == state.shape[:1]

        return outputs


class BasicShapedRewardNet(ShapedRewardNet):
    """Shaped reward net based on MLPs.

    This is just a very simple convenience class for instantiating a BasicRewardNet
    and a BasicPotentialShaping and wrapping them inside a ShapedRewardNet.
    Mainly exists for backwards compatibility after
    https://github.com/HumanCompatibleAI/imitation/pull/311
    to keep the scripts working.

    TODO(ejnnr): if we ever modify AIRL so that it takes in a RewardNet instance
        directly (instead of a class and kwargs) and instead instantiate the
        RewardNet inside the scripts, then it probably makes sense to get rid
        of this class.

    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        reward_hid_sizes: Sequence[int] = (32,),
        potential_hid_sizes: Sequence[int] = (32, 32),
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        discount_factor: float = 0.99,
    ):
        """Builds a simple shaped reward network.

        Args:
          observation_space: The observation space.
          action_space: The action space.
          reward_hid_sizes: sequence of widths for the hidden layers
            of the base reward MLP.
          potential_hid_sizes: sequence of widths for the hidden layers
            of the potential MLP.
          use_state: should the current state be included as an input to the reward MLP?
          use_action: should the current action be included as an input
            to the reward MLP?
          use_next_state: should the next state be included as an input
            to the reward MLP?
          use_done: should the "done" flag be included as an input to the reward MLP?
          discount_factor: discount factor for the potential shaping.
        """
        base_reward_net = BasicRewardNet(
            observation_space=observation_space,
            action_space=action_space,
            use_state=use_state,
            use_action=use_action,
            use_next_state=use_next_state,
            use_done=use_done,
            hid_sizes=reward_hid_sizes,
        )

        potential_net = BasicPotentialMLP(
            observation_space=observation_space, hid_sizes=potential_hid_sizes
        )

        super().__init__(
            observation_space,
            action_space,
            base_reward_net,
            potential_net,
            discount_factor=discount_factor,
        )


class BasicPotentialMLP(nn.Module):
    """Simple implementation of a potential using an MLP."""

    def __init__(self, observation_space: gym.Space, hid_sizes: Iterable[int]):
        """Initialize the potential.

        Args:
            observation_space (gym.Space): observation space of the environment.
            hid_sizes (Iterable[int]): widths of the hidden layers of the MLP.
        """
        super().__init__()
        potential_in_size = preprocessing.get_flattened_obs_dim(observation_space)
        self._potential_net = networks.build_mlp(
            in_size=potential_in_size,
            hid_sizes=hid_sizes,
            squeeze_output=True,
            flatten_input=True,
        )

    def forward(self, state: th.Tensor) -> th.Tensor:
        return self._potential_net(state)
