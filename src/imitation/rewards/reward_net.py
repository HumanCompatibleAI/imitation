"""Constructs deep network reward models."""

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence, Tuple

import gym
import tensorflow as tf

from imitation.util import networks, serialize


class RewardNet(serialize.Serializable, ABC):
    """Abstract reward network.

    This class assumes that the caller will set the default TensorFlow Session
    and initialize the network's variables.

    Attributes:
      observation_space: The observation space of `obs_ph` and `next_obs_ph`.
      action_space: The action space of `act_ph`.
      obs_ph (tf.Tensor): previous observation placeholder.
      act_ph (tf.Tensor): action placeholder.
      next_obs_ph (tf.Tensor): next observation placeholder.
      _params (dict): parameters to serialize in `save`, used as keyword
          arguments for constructor by `load`.
      _layers (dict): CheckpointableBase objects, e.g. a TensorFlow layer,
          saved by `save` and restored by `load`.
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

        self.observation_space = observation_space
        self.action_space = action_space
        self.scale = scale

        phs, inps = networks.build_inputs(observation_space, action_space, scale)
        self.obs_ph, self.act_ph, self.next_obs_ph, self.done_ph = phs
        self.obs_inp, self.act_inp, self.next_obs_inp, self.done_inp = inps

        inputs = []
        inputs += [self.obs_inp] if use_state else []
        inputs += [self.act_inp] if use_action else []
        inputs += [self.next_obs_inp] if use_next_state else []
        if len(inputs) == 0:
            raise ValueError(
                "At least one of `use_state`, `use_action` and `use_next_state` "
                "must be true."
            )
        inputs += [self.done_inp] if use_done else []

        with tf.variable_scope("theta_network"):
            self._theta_output, theta_layers = self.build_theta_network(inputs)

        self._layers = theta_layers

    @property
    @abstractmethod
    def reward_output_train(self):
        """A Tensor holding the training reward associated with each timestep.

        Different concrete subclasses will require different placeholders to be
        filled to calculate this output, but for all subclasses, filling the
        following placeholders will be sufficient:

        ```
        self.obs_ph
        self.act_ph
        self.next_obs_ph
        ```

        Returns:
          tf.Tensor: A (None,) shaped Tensor holding
              the training reward associated with each timestep.
        """
        pass

    @property
    def reward_output_test(self):
        """A Tensor holding the test reward associated with each timestep.

        Note this is the reward we use for transfer learning.

        Different concrete subclasses will require different
        placeholders to be filled to calculate this output, but for all
        subclasses, filling the following placeholders will be sufficient:

        ```
        self.obs_ph
        self.act_ph
        self.next_obs_ph
        ```

        Returns:
          tf.Tensor: A (None,) shaped Tensor holding
            the test reward associated with each timestep.
        """
        return self._theta_output

    @abstractmethod
    def build_theta_network(
        self, inputs: Sequence[tf.Tensor]
    ) -> Tuple[tf.Tensor, networks.LayersDict]:
        """Builds the test reward network.

        The output of the network is the same as the reward used for transfer
        learning, and is the Tensor returned by `self.reward_output_test()`.

        Args:
          obs_input: The observation input. Its shape is
              `((None,) + self.env.observation_space.shape)`.
          act_input: The action input. Its shape is
              `((None,) + self.env.action_space.shape)`. The None dimension is
              expected to be the same as None dimension from `obs_input`.

        Returns:
          A tuple (theta_output, layers) where
            * theta_output is a reward prediction for each of the inputs,
              of shape `(None,)`;
            * layers is a dictionary mapping to individual checkpointable layers.
        """
        pass

    def build_summaries(self):
        tf.summary.histogram("train_reward", self.reward_output_train)
        tf.summary.histogram("test_reward", self.reward_output_test)


class RewardNetShaped(RewardNet):
    """Abstract reward network with a phi network to shape training reward.

    This RewardNet formulation matches Equation (4) in the AIRL paper.
    Note that the experiments in Table 2 of the same paper showed shaped
    training rewards to be inferior to an unshaped training rewards in
    a Pendulum environment imitation learning task (and maybe HalfCheetah).
    (See original implementation of Pendulum experiment's reward function at
    https://github.com/justinjfu/inverse_rl/blob/master/inverse_rl/models/imitation_learning.py#L374)

    To make a concrete subclass, implement `build_phi_network()` and
    `build_theta_network()`.
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

        with tf.variable_scope("phi_network"):
            res = self.build_phi_network(self.obs_inp, self.next_obs_inp)
            # end_potential is the potential when the episode terminates.
            if discount_factor == 1.0:
                # If undiscounted, terminal state must have potential 0.
                end_potential = tf.constant(0.0)
            else:
                # Otherwise, it can be arbitrary, so make a trainable variable.
                end_potential = tf.Variable(
                    name="end_phi", shape=(), dtype=tf.float32, initial_value=0.0
                )
                self._layers.update(end_potential=end_potential)
            self._old_shaping_output, self._new_shaping_output, phi_layers = res
            self._layers.update(**phi_layers)

        with tf.variable_scope("f_network"):
            new_shaping = (
                self.done_inp * end_potential
                + (1 - self.done_inp) * self._new_shaping_output
            )
            self._shaped_reward_output = (
                self._theta_output
                + self._discount_factor * new_shaping
                - self._old_shaping_output
            )

    @property
    def reward_output_train(self):
        """A Tensor holding the (shaped) training reward of each timestep.

        Requires the following placeholders to be filled:

        ```
        self.obs_ph
        self.act_ph
        self.next_obs_ph
        ```

        Returns:
          tf.Tensor: A (None,) shaped Tensor holding
              the training reward associated with each timestep.
        """
        return self._shaped_reward_output

    @abstractmethod
    def build_phi_network(
        self, obs_input: tf.Tensor, next_obs_input: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, networks.LayersDict]:
        """Build the reward shaping network (disentangles dynamics from reward).

        XXX: We could potentially make it easier on the subclasser by requiring
        only one input. ie build_phi_network(obs_input). Later in
        _build_f_network, I could stack Tensors of old and new observations,
        pass them simulatenously through the network, and then unstack the
        outputs. Another way to do this would be to pass in a single
        rank 3 obs_input with shape `(2, None) + self.env.observation_space`.

        Args:
          obs_input: The old observations (corresponding to the state at which
              the current action is made). The shape of this Tensor should be
              `(None,) + self.env.observation_space.shape`.
          next_obs_input: The new observations (corresponding to the state that we
              transition to after this state-action pair.

        Returns:
          A tuple (old_shaping_output, new_shaping_output, layers) where
            * old_shaping_output is a reward shaping prediction for each of the old
              observation inputs, with shape `(None,)`.
            * new_shaping_output is a reward shaping prediction for each of the new
              observation inputs, with shape `(None,)`.
            * layers is a dictionary mapping to individual checkpointable layers.
        """
        pass

    def build_summaries(self):
        super().build_summaries()
        tf.summary.histogram("shaping_old", self._old_shaping_output)
        tf.summary.histogram("shaping_new", self._new_shaping_output)


class BasicRewardNet(RewardNet, serialize.LayersSerializable):
    """An unshaped reward net with simple, default settings.

    Intended to match the reward network trained for the original AIRL pendulum
    experiments. Right now it has a linear function approximator for the theta network,
    not sure if this is what I want.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        theta_units: Optional[Iterable[int]] = None,
        theta_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Builds a simple reward network.

        Args:
          observation_space: The observation space.
          action_space: The action space.
          theta_units: Number of hidden units at each layer of the feedforward
              reward network theta.
          theta_kwargs: Arguments passed to `build_basic_theta_network`.
          kwargs: Passed through to RewardNet.
        """
        params = locals()
        del params["kwargs"]
        params.update(kwargs)

        self.theta_units = theta_units
        self.theta_kwargs = theta_kwargs or {}
        RewardNet.__init__(self, observation_space, action_space, **kwargs)
        serialize.LayersSerializable.__init__(**params, layers=self._layers)

    def build_theta_network(self, inputs: Sequence[tf.Tensor]):
        return networks.build_and_apply_mlp(
            inputs, self.theta_units, **self.theta_kwargs
        )

    @property
    def reward_output_train(self):
        """Training reward is the same as the test reward, since no shaping."""
        return self.reward_output_test


def build_basic_phi_network(
    hid_sizes: Optional[Iterable[int]],
    obs_input: tf.Tensor,
    next_obs_input: tf.Tensor,
    **kwargs: dict,
):
    """Builds a potential network depending on specified observation.

    Arguments:
      hid_sizes: Number of units at each hidden layer. Default is (32, 32).
      obs_input: Previous observation.
      next_obs_input: Next observation.
      **kwargs: Passed through to `util.build_mlp`.

    Returns:
      Tuple[tf.Tensor, tf.Tensor]: potential for the old and new observations.
    """
    if hid_sizes is None:
        hid_sizes = (32, 32)

    with tf.variable_scope("phi", reuse=tf.AUTO_REUSE):
        old_o = tf.layers.flatten(obs_input)
        new_o = tf.layers.flatten(next_obs_input)

        # Weight share, just with different inputs old_o and new_o
        phi_mlp = networks.build_mlp(hid_sizes=hid_sizes, name="shaping", **kwargs)
        old_shaping_output = networks.sequential(old_o, phi_mlp)
        new_shaping_output = networks.sequential(new_o, phi_mlp)

    return old_shaping_output, new_shaping_output, phi_mlp


class BasicShapedRewardNet(RewardNetShaped, serialize.LayersSerializable):
    """A shaped reward network with simple, default settings.

    With default parameters this RewardNet has two hidden layers [32, 32]
    for the theta network and reward shaping phi network.

    This network is feed-forward and flattens inputs, so is a poor choice for
    training on pixel observations.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        theta_units: Optional[Iterable[int]] = None,
        theta_kwargs: Optional[dict] = None,
        phi_units: Optional[Iterable[int]] = None,
        phi_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Builds a simple shaped reward network.

        Args:
          observation_space: The observation space.
          action_space: The action space.
          theta_units: Number of hidden units at each layer of the feedforward
              reward network theta.
          theta_kwargs: Arguments passed to `build_basic_theta_network`.
          phi_units: Number of hidden units at each layer of the feedforward
              potential network phi.
          phi_kwargs: Arguments passed to `build_basic_phi_network`.
          kwargs: Passed through to `RewardNetShaped`.
        """
        params = locals()
        del params["kwargs"]
        params.update(kwargs)

        self.theta_units = theta_units
        self.phi_units = phi_units
        self.theta_kwargs = theta_kwargs or {}
        self.phi_kwargs = phi_kwargs or {}
        RewardNetShaped.__init__(
            self, observation_space, action_space, **kwargs,
        )
        serialize.LayersSerializable.__init__(**params, layers=self._layers)

    def build_theta_network(self, inputs):
        return networks.build_and_apply_mlp(
            inputs, self.theta_units, **self.theta_kwargs
        )

    def build_phi_network(self, obs_input, next_obs_input):
        return build_basic_phi_network(
            self.phi_units, obs_input, next_obs_input, **self.phi_kwargs
        )
