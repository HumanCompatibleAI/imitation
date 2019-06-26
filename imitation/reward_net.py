"""Constructs deep network reward models."""

from abc import ABC, abstractmethod

import tensorflow as tf

import imitation.util as util


class RewardNet(ABC):
  """Abstract reward network.

  This class assumes that the caller will set the default TensorFlow Session
  and initialize the network's variables.

  Attributes:
    env (gym.Env): the environment we are predicting reward for.
    old_obs_ph (tf.Tensor): previous observation placeholder.
    act_ph (tf.Tensor): action placeholder.
    new_obs_ph (tf.Tensor): next observation placeholder.
  """

  def __init__(self, env):
    """Builds a reward network.

    Args:
        env (gym.Env or str): The environment we are predicting reward for.
    """

    self.env = util.maybe_load_env(env)

    phs = util.build_placeholders(self.env, True)
    self.old_obs_ph, self.act_ph, self.new_obs_ph = phs

    with tf.variable_scope("theta_network"):
      self._theta_output = self.build_theta_network(
          self.old_obs_ph, self.act_ph)
      # TODO: assert that all the outputs above have shape (None,).

  @property
  @abstractmethod
  def reward_output_train(self):
    """A Tensor holding the training reward associated with each timestep.

    Different concrete subclasses will require different placeholders to be
    filled to calculate this output, but for all subclasses, filling the
    following placeholders will be sufficient:

    ```
    self.old_obs_ph
    self.act_ph
    self.new_obs_ph
    ```

    Returns:
      _reward_output_train (tf.Tensor) A (None,): shaped Tensor holding
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
    self.old_obs_ph
    self.act_ph
    self.new_obs_ph
    ```

    Returns:
      _reward_output_test (tf.Tensor) A (None,): shaped Tensor holding
        the test reward associated with each timestep.
    """
    return self._theta_output

  @abstractmethod
  def build_theta_network(self, obs_input, act_input):
    """Builds the test reward network.

    The output of the network is the same as the reward used for transfer
    learning, and is the Tensor returned by `self.reward_output_test()`.

    Args:
      obs_input (tf.Tensor): The observation input. Its shape is
          `((None,) + self.env.observation_space.shape)`.
      act_input (tf.Tensor): The action input. Its shape is
          `((None,) + self.env.action_space.shape)`. The None dimension is
          expected to be the same as None dimension from `obs_input`.

    Returns:
      theta_output (tf.Tensor): A reward prediction for each of the
          inputs. The shape is `(None,)`.
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

  def __init__(self, env, *, discount_factor=0.9, **kwargs):
    super().__init__(env, **kwargs)
    self._discount_factor = discount_factor

    with tf.variable_scope("phi_network"):
      res = self.build_phi_network(self.old_obs_ph, self.new_obs_ph)
      self._old_shaping_output, self._new_shaping_output = res

    with tf.variable_scope("f_network"):
      self._shaped_reward_output = (
        self._theta_output
        + self._discount_factor * self._new_shaping_output
        - self._old_shaping_output)

  @property
  def reward_output_train(self):
    """A Tensor holding the (shaped) training reward of each timestep.

    Requires the following placeholders to be filled:

    ```
    self.old_obs_ph
    self.act_ph
    self.new_obs_ph
    ```

    Returns:
      _reward_output_train (tf.Tensor) A (None,): shaped Tensor holding
          the training reward associated with each timestep.
    """
    return self._shaped_reward_output

  @abstractmethod
  def build_phi_network(self, old_obs_input, new_obs_input):
    """Build the reward shaping network (disentangles dynamics from reward).

    XXX: We could potentially make it easier on the subclasser by requiring
    only one input. ie build_phi_network(obs_input). Later in
    _build_f_network, I could stack Tensors of old and new observations,
    pass them simulatenously through the network, and then unstack the
    outputs. Another way to do this would be to pass in a single
    rank 3 obs_input with shape `(2, None) + self.env.observation_space`.

    Args:
      old_obs_input (tf.Tensor): The old observations (corresponding to the
          state at which the current action is made). The shape of this
          Tensor should be `(None,) + self.env.observation_space.shape`.
      new_obs_input (tf.Tensor): The new observations (corresponding to
          the state that we transition to after this state-action pair.

    Returns:
      old_shaping_output (tf.Tensor): A reward shaping prediction for
          each of the old observation inputs. Has shape `(None,)`.
      new_shaping_output (tf.Tensor): A reward shaping prediction for
          each of the new observation inputs. Has shape `(None,)`
    """
    pass

  def build_summaries(self):
    super().build_summaries()
    tf.summary.histogram("shaping_old", self._old_shaping_output)
    tf.summary.histogram("shaping_new", self._new_shaping_output)


def build_basic_theta_network(hid_sizes, obs_space, act_space,
                              old_obs_input, new_obs_input, act_input):
  """Builds a reward network depending on specified observations and actions.

  All specified inputs will be preprocessed and then concatenated. If all
  inputs are specified, then it will be a :math:`R(o,a,o')` network.
  Conversely, if `new_obs_input` and `act_input` are both set to `None`, it
  will depend just on the current observation: :math:`R(o)`.

  Arguments:
    hid_sizes (Optional[List[int]]): Number of units at each hidden layer.
    obs_space (gym.Space): Observation space.
    act_space (gym.Space): Action space.
    old_obs_input (Optional[tf.Tensor]): Previous observation.
    new_obs_input (Optional[tf.Tensor]): Next observation.
    act_input (tf.Tensor): Action.

  Returns:
    tf.Tensor: Predicted reward.

  Raises:
    ValueError: If all of old_obs_input, new_obs_input and act_input are None.
  """
  if hid_sizes is None:
    hid_sizes = []

  with tf.variable_scope("theta"):
    inputs = [(old_obs_input, obs_space),
              (act_input, act_space),
              (new_obs_input, obs_space)]
    inputs = [(input, space) for input, space in inputs if input is not None]
    if len(inputs) == 0:
      raise ValueError("Must specify at least one input")

    inputs = [util.flat(input, space.shape) for input, space in inputs]
    inputs = tf.concat(inputs, axis=1)
    theta_output = util.apply_ff(inputs, hid_sizes=hid_sizes)

    return theta_output


class BasicRewardNet(RewardNet):
  """An unshaped reward net with simple, default settings.

  Intended to match the reward network trained for
  the original AIRL pendulum experiments. Right now it has a linear function
  approximator for the theta network, not sure if this is what I want.
  """

  def __init__(self, env, *, state_only=False, theta_units=None, **kwargs):
    """Builds a simple reward network.

    Args:
      env (gym.Env or str): The environment we are predicting reward for.
      state_only (bool): If True, then ignore the action when predicting
          and training the reward network phi.
      theta_units (List[int]): number of hidden units at each layer of
          the feedforward theta network.
    """
    self.state_only = state_only
    self.theta_units = theta_units
    super().__init__(env, **kwargs)

  def build_theta_network(self, obs_input, act_input):
    act_or_none = None if self.state_only else act_input
    return build_basic_theta_network(self.theta_units,
                                     self.env.observation_space,
                                     self.env.action_space,
                                     old_obs_input=obs_input,
                                     act_input=act_or_none,
                                     new_obs_input=None)

  @property
  def reward_output_train(self):
    """Training reward is the same as the test reward, since no shaping."""
    return self.reward_output_test


def build_basic_phi_network(hid_sizes, obs_space, old_obs_input, new_obs_input):
  """Builds a potential network depending on specified observation.

  Arguments:
    hid_sizes (Optional[List[int]]): Number of units at each hidden layer.
    obs_space (gym.Space): Observation space.
    old_obs_input (Optional[tf.Tensor]): Previous observation.
    new_obs_input (Optional[tf.Tensor]): Next observation.

  Returns:
    Tuple[tf.Tensor]: potential for the old and new observations.
  """
  if hid_sizes is None:
    hid_sizes = [32, 32]

  with tf.variable_scope("phi", reuse=tf.AUTO_REUSE):
    old_o = util.flat(old_obs_input, obs_space.shape)
    new_o = util.flat(new_obs_input, obs_space.shape)

    # Weight share, just with different inputs old_o and new_o
    old_shaping_output = tf.identity(
        util.apply_ff(old_o, hid_sizes=hid_sizes),
        name="old_shaping_output")
    new_shaping_output = tf.identity(
        util.apply_ff(new_o, hid_sizes=hid_sizes),
        name="new_shaping_output")

  return old_shaping_output, new_shaping_output


class BasicShapedRewardNet(RewardNetShaped):
  """A shaped reward network with simple, default settings.

  With default parameters this RewardNet has two hidden layers [32, 32]
  for the reward shaping phi network, and a linear function approximator
  for the theta network. These settings match the network architectures for
  continuous control experiments described in Appendix D.1 of the
  AIRL paper.

  This network flattens inputs. So it isn't suitable for training on
  pixel observations.
  """

  def __init__(self, env, *, state_only=False,
               theta_units=None, phi_units=None, **kwargs):
    """Builds a simple shaped reward network.

    Args:
      env (gym.Env or str): The environment we are predicitng reward for.
      units (int): The number of hidden units in each of the two layers
          of the feed forward phi network.
      state_only (bool): If True, then ignore the action when predicting
          and training the reward network phi.
      theta_units (List[int]): number of hidden units at each layer of
          the feedforward theta network.
      phi_units: (List[int] number of hidden units at each layer of
          the feedforward phi network.
      discount_factor (float): A number in the range [0, 1].
    """
    self.state_only = state_only
    self.theta_units = theta_units
    self.phi_units = phi_units
    super().__init__(env, **kwargs)

  def build_theta_network(self, obs_input, act_input):
    act_or_none = None if self.state_only else act_input
    return build_basic_theta_network(self.theta_units,
                                     self.env.observation_space,
                                     self.env.action_space,
                                     old_obs_input=obs_input,
                                     act_input=act_or_none,
                                     new_obs_input=None)

  def build_phi_network(self, old_obs_input, new_obs_input):
    return build_basic_phi_network(self.phi_units, self.env.observation_space,
                                   old_obs_input, new_obs_input)
