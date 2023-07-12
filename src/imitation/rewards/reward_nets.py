"""Constructs deep network reward models."""

import abc
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Type, cast

import gym
import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common import preprocessing
from torch import nn

from imitation.util import networks, util


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
            observation_space: the observation space of the environment
            action_space: the action space of the environment
            normalize_images: whether to automatically normalize
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
    ) -> th.Tensor:
        """Compute rewards for a batch of transitions and keep gradients."""

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Preprocess a batch of input transitions and convert it to PyTorch tensors.

        The output of this function is suitable for its forward pass,
        so a typical usage would be ``model(*model.preprocess(transitions))``.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            action: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.

        Returns:
            Preprocessed transitions: a Tuple of tensors containing
            observations, actions, next observations and dones.
        """
        state_th = util.safe_to_tensor(state).to(self.device)
        action_th = util.safe_to_tensor(action).to(self.device)
        next_state_th = util.safe_to_tensor(next_state).to(self.device)
        done_th = util.safe_to_tensor(done).to(self.device)

        del state, action, next_state, done  # unused

        # preprocess
        # we only support array spaces, so we cast
        # the observation to torch tensors.
        state_th = cast(
            th.Tensor,
            preprocessing.preprocess_obs(
                state_th,
                self.observation_space,
                self.normalize_images,
            ),
        )
        action_th = cast(
            th.Tensor,
            preprocessing.preprocess_obs(
                action_th,
                self.action_space,
                self.normalize_images,
            ),
        )
        next_state_th = cast(
            th.Tensor,
            preprocessing.preprocess_obs(
                next_state_th,
                self.observation_space,
                self.normalize_images,
            ),
        )
        done_th = done_th.to(th.float32)

        n_gen = len(state_th)
        assert state_th.shape == next_state_th.shape
        assert len(action_th) == n_gen

        return state_th, action_th, next_state_th, done_th

    def predict_th(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> th.Tensor:
        """Compute th.Tensor rewards for a batch of transitions without gradients.

        Preprocesses the inputs, output th.Tensor reward arrays.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.

        Returns:
            Computed th.Tensor rewards of shape `(batch_size,`).
        """
        with networks.evaluating(self):
            # switch to eval mode (affecting normalization, dropout, etc)

            state_th, action_th, next_state_th, done_th = self.preprocess(
                state,
                action,
                next_state,
                done,
            )
            with th.no_grad():
                rew_th = self(state_th, action_th, next_state_th, done_th)

            assert rew_th.shape == state.shape[:1]  # TODO do we or don't we support
            # state spaces with more than one axis?
            return rew_th

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Compute rewards for a batch of transitions without gradients.

        Converting th.Tensor rewards from `predict_th` to NumPy arrays.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.

        Returns:
            Computed rewards of shape `(batch_size,)`.
        """
        rew_th = self.predict_th(state, action, next_state, done)
        return rew_th.detach().cpu().numpy().flatten()

    def predict_processed(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute the processed rewards for a batch of transitions without gradients.

        Defaults to calling `predict`. Subclasses can override this to normalize or
        otherwise modify the rewards in ways that may help RL training or other
        applications of the reward function.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            kwargs: additional kwargs may be passed to change the functionality of
                subclasses.

        Returns:
            Computed processed rewards of shape `(batch_size,`).
        """
        del kwargs
        return self.predict(state, action, next_state, done)

    @property
    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        try:
            first_param = next(self.parameters())
            return first_param.device
        except StopIteration:
            # if the model has no parameters, we use the CPU
            return th.device("cpu")

    @property
    def dtype(self) -> th.dtype:
        """Heuristic to determine dtype of module."""
        try:
            first_param = next(self.parameters())
            return first_param.dtype
        except StopIteration:
            # if the model has no parameters, default to float32
            return th.get_default_dtype()


class RewardNetWrapper(RewardNet):
    """Abstract class representing a wrapper modifying a ``RewardNet``'s functionality.

    In general ``RewardNetWrapper``s should either subclass ``ForwardWrapper``
    or ``PredictProcessedWrapper``.
    """

    def __init__(
        self,
        base: RewardNet,
    ):
        """Initialize a RewardNet wrapper.

        Args:
            base: the base RewardNet to wrap.
        """
        super().__init__(
            base.observation_space,
            base.action_space,
            base.normalize_images,
        )
        self._base = base

    @property
    def base(self) -> RewardNet:
        return self._base

    @property
    def device(self) -> th.device:
        __doc__ = super().device.__doc__  # noqa: F841
        return self.base.device

    @property
    def dtype(self) -> th.dtype:
        __doc__ = super().dtype.__doc__  # noqa: F841
        return self.base.dtype

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        __doc__ = super().preprocess.__doc__  # noqa: F841
        return self.base.preprocess(state, action, next_state, done)


class ForwardWrapper(RewardNetWrapper):
    """An abstract RewardNetWrapper that changes the behavior of forward.

    Note that all forward wrappers must be placed before all
    predict processed wrappers.
    """

    def __init__(
        self,
        base: RewardNet,
    ):
        """Create a forward wrapper.

        Args:
            base: The base reward network

        Raises:
            ValueError: if the base network is a `PredictProcessedWrapper`.
        """
        super().__init__(base)
        if isinstance(base, PredictProcessedWrapper):
            # Doing this could cause confusing errors like normalization
            # not being applied.
            raise ValueError(
                "ForwardWrapper cannot be applied on top of PredictProcessedWrapper!",
            )


class PredictProcessedWrapper(RewardNetWrapper):
    """An abstract RewardNetWrapper that changes the behavior of predict_processed.

    Subclasses should override `predict_processed`. Implementations
    should pass along `kwargs` to the `base` reward net's `predict_processed` method.

    Note: The wrapper will default to forwarding calls to `device`, `forward`,
        `preprocess` and `predict` to the base reward net unless
        explicitly overridden in a subclass.
    """

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Compute rewards for a batch of transitions and keep gradients."""
        return self.base.forward(state, action, next_state, done)

    @abc.abstractmethod
    def predict_processed(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Predict processed must be overridden in subclasses."""

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        __doc__ = super().predict.__doc__  # noqa: F841
        return self.base.predict(state, action, next_state, done)

    def predict_th(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> th.Tensor:
        __doc__ = super().predict_th.__doc__  # noqa: F841
        return self.base.predict_th(state, action, next_state, done)


class RewardNetWithVariance(RewardNet):
    """A reward net that keeps track of its epistemic uncertainty through variance."""

    @abc.abstractmethod
    def predict_reward_moments(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the mean and variance of the reward distribution.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            **kwargs: may modify the behavior of subclasses

        Returns:
            * Estimated reward mean of shape `(batch_size,)`.
            * Estimated reward variance of shape `(batch_size,)`. # noqa: DAR202
        """


class BasicRewardNet(RewardNet):
    """MLP that takes as input the state, action, next state and done flag.

    These inputs are flattened and then concatenated to one another. Each input
    can enabled or disabled by the `use_*` constructor keyword arguments.
    """

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
            kwargs: passed straight through to `build_mlp`.
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

        full_build_mlp_kwargs: Dict[str, Any] = {
            "hid_sizes": (32, 32),
            **kwargs,
            # we do not want the values below to be overridden
            "in_size": combined_size,
            "out_size": 1,
            "squeeze_output": True,
        }

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


class CnnRewardNet(RewardNet):
    """CNN that takes as input the state, action, next state and done flag.

    Inputs are boosted to tensors with channel, height, and width dimensions, and then
    concatenated. Image inputs are assumed to be in (h,w,c) format, unless the argument
    hwc_format=False is passed in. Each input can be enabled or disabled by the `use_*`
    constructor keyword arguments, but either `use_state` or `use_next_state` must be
    True.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        hwc_format: bool = True,
        **kwargs,
    ):
        """Builds reward CNN.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: Should the current state be included as an input to the CNN?
            use_action: Should the current action be included as an input to the CNN?
            use_next_state: Should the next state be included as an input to the CNN?
            use_done: Should the "done" flag be included as an input to the CNN?
            hwc_format: Are image inputs in (h,w,c) format (True), or (c,h,w) (False)?
                If hwc_format is False, image inputs are not transposed.
            kwargs: Passed straight through to `build_cnn`.

        Raises:
            ValueError: if observation or action space is not easily massaged into a
                CNN input.
        """
        super().__init__(observation_space, action_space)
        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        self.use_done = use_done
        self.hwc_format = hwc_format

        if not (self.use_state or self.use_next_state):
            raise ValueError("CnnRewardNet must take current or next state as input.")

        if not preprocessing.is_image_space(observation_space):
            raise ValueError(
                "CnnRewardNet requires observations to be images.",
            )
        if self.use_action and not isinstance(action_space, spaces.Discrete):
            raise ValueError(
                "CnnRewardNet can only use Discrete action spaces.",
            )

        input_size = 0
        output_size = 1

        if self.use_state:
            input_size += self.get_num_channels_obs(observation_space)

        if self.use_action:
            output_size = action_space.n

        if self.use_next_state:
            input_size += self.get_num_channels_obs(observation_space)

        if self.use_done:
            output_size *= 2

        full_build_cnn_kwargs: Dict[str, Any] = {
            "hid_channels": (32, 32),
            **kwargs,
            # we do not want the values below to be overridden
            "in_channels": input_size,
            "out_size": output_size,
            "squeeze_output": output_size == 1,
        }

        self.cnn = networks.build_cnn(**full_build_cnn_kwargs)

    def get_num_channels_obs(self, space: spaces.Box) -> int:
        """Gets number of channels for the observation."""
        return space.shape[-1] if self.hwc_format else space.shape[0]

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Computes rewardNet value on input state, action, next_state, and done flag.

        Takes inputs that will be used, transposes image states to (c,h,w) format if
        needed, reshapes inputs to have compatible dimensions, concatenates them, and
        inputs them into the CNN.

        Args:
            state: current state.
            action: current action.
            next_state: next state.
            done: flag for whether the episode is over.

        Returns:
            th.Tensor: reward of the transition.
        """
        inputs = []
        if self.use_state:
            state_ = cnn_transpose(state) if self.hwc_format else state
            inputs.append(state_)
        if self.use_next_state:
            next_state_ = cnn_transpose(next_state) if self.hwc_format else next_state
            inputs.append(next_state_)

        inputs_concat = th.cat(inputs, dim=1)
        outputs = self.cnn(inputs_concat)
        if self.use_action and not self.use_done:
            # for discrete action spaces, action is passed to forward as a one-hot
            # vector.
            rewards = th.sum(outputs * action, dim=1)
        elif self.use_action and self.use_done:
            # here, we double the size of the one-hot vector, where the first entries
            # are for done=False and the second are for done=True.
            action_done_false = action * (1 - done[:, None])
            action_done_true = action * done[:, None]
            full_acts = th.cat((action_done_false, action_done_true), dim=1)
            rewards = th.sum(outputs * full_acts, dim=1)
        elif not self.use_action and self.use_done:
            # here we turn done into a one-hot vector.
            dones_binary = done.long()
            dones_one_hot = nn.functional.one_hot(dones_binary, num_classes=2)
            rewards = th.sum(outputs * dones_one_hot, dim=1)
        else:
            rewards = outputs
        return rewards


def cnn_transpose(tens: th.Tensor) -> th.Tensor:
    """Transpose a (b,h,w,c)-formatted tensor to (b,c,h,w) format."""
    if len(tens.shape) == 4:
        return th.permute(tens, (0, 3, 1, 2))
    else:
        raise ValueError(
            f"Invalid input: len(tens.shape) = {len(tens.shape)} != 4.",
        )


class NormalizedRewardNet(PredictProcessedWrapper):
    """A reward net that normalizes the output of its base network."""

    def __init__(
        self,
        base: RewardNet,
        normalize_output_layer: Type[networks.BaseNorm],
    ):
        """Initialize the NormalizedRewardNet.

        Args:
            base: a base RewardNet
            normalize_output_layer: The class to use to normalize rewards. This
                can be any nn.Module that preserves the shape; e.g. `nn.Identity`,
                `nn.LayerNorm`, or `networks.RunningNorm`.
        """
        # Note(yawen): by default, the reward output is squeezed to produce
        # tensors with (N,) shape for predict_processed. This works for
        # `networks.RunningNorm`, but not for `nn.BatchNorm1d` that requires
        # shape of (N,C).
        super().__init__(base=base)
        # Assuming reward is scalar, norm layer should be initialized with shape (1,).
        self.normalize_output_layer = normalize_output_layer(1)

    def predict_processed(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        update_stats: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Compute normalized rewards for a batch of transitions without gradients.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            update_stats: Whether to update the running stats of the normalization
                layer.
            **kwargs: kwargs passed to base predict_processed call.

        Returns:
            Computed normalized rewards of shape `(batch_size,`).
        """
        with networks.evaluating(self):
            # switch to eval mode (affecting normalization, dropout, etc)
            rew_th = th.tensor(
                self.base.predict_processed(state, action, next_state, done, **kwargs),
                device=self.device,
            )
            rew = self.normalize_output_layer(rew_th).detach().cpu().numpy().flatten()
        if update_stats:
            with th.no_grad():
                self.normalize_output_layer.update_stats(rew_th)
        assert rew.shape == state.shape[:1]
        return rew


class ShapedRewardNet(ForwardWrapper):
    """A RewardNet consisting of a base network and a potential shaping."""

    def __init__(
        self,
        base: RewardNet,
        potential: Callable[[th.Tensor], th.Tensor],
        discount_factor: float,
    ):
        """Setup a ShapedRewardNet instance.

        Args:
            base: the base reward net to which the potential shaping
                will be added. Shaping must be applied directly to the raw reward net.
                See error below.
            potential: A callable which takes
                a batch of states (as a PyTorch tensor) and returns a batch of
                potentials for these states. If this is a PyTorch Module, it becomes
                a submodule of the ShapedRewardNet instance.
            discount_factor: discount factor to use for the potential shaping.
        """
        super().__init__(
            base=base,
        )
        self.potential = potential
        self.discount_factor = discount_factor

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
        # NOTE(ejnnr): We fix the potential of terminal states to zero, which is
        # necessary for valid potential shaping in a variable-length horizon setting.
        #
        # In more detail: variable-length episodes are usually modeled
        # as infinite-length episodes where we transition to a terminal state
        # in which we then remain forever. The transition to this final
        # state contributes gamma * Phi(s_T) - Phi(s_{T - 1}) to the returns,
        # where Phi is the potential and s_T the final state. But on every step
        # afterwards, the potential shaping leads to a reward of (gamma - 1) * Phi(s_T).
        # The discounted series of these rewards, which is added to the return,
        # is gamma / (1 - gamma) times this reward, i.e. just -gamma * Phi(s_T).
        # This cancels the contribution of the final state to the last "real"
        # transition, so instead of computing the infinite series, we can
        # equivalently fix the final potential to zero without loss of generality.
        # Not fixing the final potential to zero and also not adding this infinite
        # series of remaining potential shapings can lead to reward shaping
        # that does not preserve the optimal policy if the episodes have variable
        # length!
        new_shaping = (1 - done.float()) * new_shaping_output
        final_rew = (
            base_reward_net_output
            + self.discount_factor * new_shaping
            - old_shaping_output
        )
        assert final_rew.shape == state.shape[:1]
        return final_rew


class BasicShapedRewardNet(ShapedRewardNet):
    """Shaped reward net based on MLPs.

    This is just a very simple convenience class for instantiating a BasicRewardNet
    and a BasicPotentialMLP and wrapping them inside a ShapedRewardNet.
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
        **kwargs,
    ):
        """Builds a simple shaped reward network.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            reward_hid_sizes: sequence of widths for the hidden layers
                of the base reward MLP.
            potential_hid_sizes: sequence of widths for the hidden layers
                of the potential MLP.
            use_state: should the current state be included as an input
                to the reward MLP?
            use_action: should the current action be included as an input
                to the reward MLP?
            use_next_state: should the next state be included as an input
                to the reward MLP?
            use_done: should the "done" flag be included as an input to the reward MLP?
            discount_factor: discount factor for the potential shaping.
            kwargs: passed straight through to `BasicRewardNet` and `BasicPotentialMLP`.
        """
        base_reward_net = BasicRewardNet(
            observation_space=observation_space,
            action_space=action_space,
            use_state=use_state,
            use_action=use_action,
            use_next_state=use_next_state,
            use_done=use_done,
            hid_sizes=reward_hid_sizes,
            **kwargs,
        )

        potential_net = BasicPotentialMLP(
            observation_space=observation_space,
            hid_sizes=potential_hid_sizes,
            **kwargs,
        )

        super().__init__(
            base_reward_net,
            potential_net,
            discount_factor=discount_factor,
        )


class BasicPotentialMLP(nn.Module):
    """Simple implementation of a potential using an MLP."""

    def __init__(
        self,
        observation_space: gym.Space,
        hid_sizes: Iterable[int],
        **kwargs,
    ):
        """Initialize the potential.

        Args:
            observation_space: observation space of the environment.
            hid_sizes: widths of the hidden layers of the MLP.
            kwargs: passed straight through to `build_mlp`.
        """
        super().__init__()
        potential_in_size = preprocessing.get_flattened_obs_dim(observation_space)
        self._potential_net = networks.build_mlp(
            in_size=potential_in_size,
            hid_sizes=hid_sizes,
            squeeze_output=True,
            flatten_input=True,
            **kwargs,
        )

    def forward(self, state: th.Tensor) -> th.Tensor:
        return self._potential_net(state)


class BasicPotentialCNN(nn.Module):
    """Simple implementation of a potential using a CNN."""

    def __init__(
        self,
        observation_space: gym.Space,
        hid_sizes: Iterable[int],
        hwc_format: bool = True,
        **kwargs,
    ):
        """Initialize the potential.

        Args:
            observation_space: observation space of the environment.
            hid_sizes: number of channels in hidden layers of the CNN.
            hwc_format: format of the observation. True if channel dimension is last,
                False if channel dimension is first.
            kwargs: passed straight through to `build_cnn`.

        Raises:
            ValueError: if observations are not images.
        """
        super().__init__()
        self.hwc_format = hwc_format
        if not preprocessing.is_image_space(observation_space):
            raise ValueError("CNN potential must be given image inputs.")
        obs_shape = observation_space.shape
        in_channels = obs_shape[-1] if self.hwc_format else obs_shape[0]
        self._potential_net = networks.build_cnn(
            in_channels=in_channels,
            hid_channels=hid_sizes,
            squeeze_output=True,
            **kwargs,
        )

    def forward(self, state: th.Tensor) -> th.Tensor:
        state_ = cnn_transpose(state) if self.hwc_format else state
        return self._potential_net(state_)


class RewardEnsemble(RewardNetWithVariance):
    """A mean ensemble of reward networks.

    A reward ensemble is made up of individual reward networks. To maintain consistency
    the "output" of a reward network will be defined as the results of its
    `predict_processed`. Thus for example the mean of the ensemble is the mean of the
    results of its members predict processed classes.
    """

    members: nn.ModuleList

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        members: Iterable[RewardNet],
    ):
        """Initialize the RewardEnsemble.

        Args:
            observation_space: the observation space of the environment
            action_space: the action space of the environment
            members: the member networks that will make up the ensemble.

        Raises:
            ValueError: if num_members is less than 1
        """
        super().__init__(observation_space, action_space)

        members = list(members)
        if len(members) < 2:
            raise ValueError("Must be at least 2 member in the ensemble.")

        self.members = nn.ModuleList(
            members,
        )

    @property
    def num_members(self):
        """The number of members in the ensemble."""
        return len(self.members)

    def predict_processed_all(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Get the results of predict processed on all of the members.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            kwargs: passed along to ensemble members.

        Returns:
            The result of predict processed for each member in the ensemble of
                shape `(batch_size, num_members)`.
        """
        batch_size = state.shape[0]
        rewards_list = [
            member.predict_processed(state, action, next_state, done, **kwargs)
            for member in self.members
        ]
        rewards: np.ndarray = np.stack(rewards_list, axis=-1)
        assert rewards.shape == (batch_size, self.num_members)
        return rewards

    @th.no_grad()
    def predict_reward_moments(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the standard deviation of the reward distribution for a batch.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            **kwargs: passed along to predict processed.

        Returns:
            * Reward mean of shape `(batch_size,)`.
            * Reward variance of shape `(batch_size,)`.
        """
        batch_size = state.shape[0]
        all_rewards = self.predict_processed_all(
            state,
            action,
            next_state,
            done,
            **kwargs,
        )
        mean_reward = all_rewards.mean(-1)
        var_reward = all_rewards.var(-1, ddof=1)
        assert mean_reward.shape == var_reward.shape == (batch_size,)
        return mean_reward, var_reward

    def forward(self, *args) -> th.Tensor:
        """The forward method of the ensemble should in general not be used directly."""
        raise NotImplementedError

    def predict_processed(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Return the mean of the ensemble members."""
        return self.predict(state, action, next_state, done, **kwargs)

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ):
        """Return the mean of the ensemble members."""
        mean, _ = self.predict_reward_moments(state, action, next_state, done, **kwargs)
        return mean


class AddSTDRewardWrapper(PredictProcessedWrapper):
    """Adds a multiple of the estimated standard deviation to mean reward."""

    base: RewardNetWithVariance

    def __init__(self, base: RewardNetWithVariance, default_alpha: float = 0.0):
        """Create a reward network that adds a multiple of the standard deviation.

        Args:
            base: A reward network that keeps track of its epistemic variance.
                This is used to compute the standard deviation.
            default_alpha: multiple of standard deviation to add to the reward mean.
                Defaults to 0.0.

        Raises:
            TypeError: if base is not an instance of RewardNetWithVariance
        """
        super().__init__(base)
        if not isinstance(base, RewardNetWithVariance):
            raise TypeError(
                "Cannot add standard deviation to reward net that "
                "is not an instance of RewardNetWithVariance!",
            )

        self.default_alpha = default_alpha

    def predict_processed(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        alpha: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute a lower/upper confidence bound on the reward without gradients.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            alpha: multiple of standard deviation to add to the reward mean. Defaults
                to the value provided at initialization.
            **kwargs: are not used

        Returns:
            Estimated lower confidence bounds on rewards of shape `(batch_size,`).
        """
        del kwargs

        if alpha is None:
            alpha = self.default_alpha

        reward_mean, reward_var = self.base.predict_reward_moments(
            state,
            action,
            next_state,
            done,
        )

        return reward_mean + alpha * np.sqrt(reward_var)
