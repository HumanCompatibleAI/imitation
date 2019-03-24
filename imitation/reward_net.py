from abc import ABC, abstractmethod

import imitation.util as util
import tensorflow as tf


class RewardNet(ABC):
    def __init__(self, env):
        """
        Reward network for Adversarial IRL.

        This class assumes that the caller will
        set the default Tensorflow Session and initialize the network's
        variables.

        Params:
          env (gym.Env or str): The environment that we are predicting reward
            for.
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
        """
        Returns a Tensor that holds the training reward associated with each
        timestep. Different concrete subclasses will require different
        placeholders to be filled to calculate this output, but for all
        subclasses, filling the following placeholders will be sufficient:

        ```
        self.old_obs_ph
        self.act_ph
        self.new_obs_ph
        ```

        Returns:
        _reward_output_train (Tensor): A (None,) shaped Tensor holding
          the training reward associated with each timestep.
        """
        pass

    @property
    def reward_output_test(self):
        """
        Returns a Tensor that holds the test reward associated with each
        timestep. (This is the reward we use for transfer learning.)
        Different concrete subclasses will require different
        placeholders to be filled to calculate this output, but for all
        subclasses, filling the following placeholders will be sufficient:

        ```
        self.old_obs_ph
        self.act_ph
        self.new_obs_ph
        ```

        Returns:
        _reward_output_test (Tensor): A (None,) shaped Tensor holding
          the test reward associated with each timestep.
        """
        return self._theta_output

    @abstractmethod
    def build_theta_network(self, obs_input, act_input):
        """
        Build the test reward network. The output of the network is
        the same as the reward we use during transfer learning, and
        is the Tensor returned by `self.reward_output_test()`.

        Params:
        obs_input (Tensor) -- The observation input. Its shape is
          `((None,) + self.env.observation_space.shape)`.
        act_input (Tensor) -- The action input. Its shape is
          `((None,) + self.env.action_space.shape)`. The None dimension is
          expected to be the same as None dimension from `obs_input`.

        Return:
        theta_output (Tensor) -- A reward prediction for each of the
          inputs. The shape is `(None,)`.
        """
        pass

    def build_summaries(self):
        tf.summary.histogram("train_reward", self.reward_output_train)
        tf.summary.histogram("test_reward", self.reward_output_test)


class RewardNetShaped(RewardNet):
    """
    Abstract subclass of RewardNet that adds a abstract phi network used to
    shape the training reward. This RewardNet formulation matches Equation (4)
    in the AIRL paper.

    Note that the experiments in Table 2 of the same paper showed
    shaped training rewards
    to be inferior to an unshaped training rewards in
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
                    self._theta_output +
                    self._discount_factor * self._new_shaping_output
                    - self._old_shaping_output)

    @property
    def reward_output_train(self):
        """
        Returns a Tensor that holds the (shaped) training reward associated
        with each timestep. Requires the following placeholders to be filled:

        ```
        self.old_obs_ph
        self.act_ph
        self.new_obs_ph
        ```

        Returns:
        _reward_output_train (Tensor): A (None,) shaped Tensor holding
          the training reward associated with each timestep.
        """
        return self._shaped_reward_output

    @abstractmethod
    def build_phi_network(self, old_obs_input, new_obs_input):
        """
        Build the reward shaping network (serves to disentangle dynamics from
        reward).

        XXX: We could potentially make it easier on the subclasser by requiring
          only one input. ie build_phi_network(obs_input). Later in
          _build_f_network, I could stack Tensors of old and new observations,
          pass them simulatenously through the network, and then unstack the
          outputs. Another way to do this would be to pass in a single
          rank 3 obs_input with shape
          `(2, None) + self.env.observation_space`.

        Params:
        old_obs_input (Tensor): The old observations (corresponding to the
          state at which the current action is made). The shape of this
          Tensor should be `(None,) + self.env.observation_space.shape`.
        new_obs_input (Tensor): The new observations (corresponding to the
          state that we transition to after this state-action pair.

        Return:
        old_shaping_output (Tensor) -- A reward shaping prediction for each of
          the old observation inputs. Has shape `(None,)` (batch size).
        new_shaping_output (Tensor) -- A reward shaping prediction for each of
          the new observation inputs. Has shape `(None,)` (batch size).
        """
        pass

    def build_summaries(self):
        super().build_summaries()
        tf.summary.histogram("shaping_old", self._old_shaping_output)
        tf.summary.histogram("shaping_new", self._new_shaping_output)


class BasicShapedRewardNet(RewardNetShaped):
    """
    A shaped reward network with default settings.
    With default parameters this RewardNet has two hidden layers [32, 32]
    for the reward shaping phi network, and a linear function approximator
    for the theta network. These settings match the network architectures for
    continuous control experiments described in Appendix D.1 of the
    AIRL paper.

    This network flattens inputs. So it isn't suitable for training on
    pixel observations.
    """

    def __init__(self, env, *, units=32, state_only=False, **kwargs):
        """
        Params:
          env (gym.Env or str): The environment that we are predicting reward
            for.
          units (int): The number of hidden units in each of the two layers of
            the feed forward phi network.
          state_only (bool): If True, then ignore the action when predicting
            and training the reward network phi.
          discount_factor (float): A number in the range [0, 1].
        """
        self.state_only = state_only
        self._units = units
        super().__init__(env, **kwargs)

    def build_theta_network(self, obs_input, act_input):
        if self.state_only:
            inputs = util.flat(obs_input)
        else:
            inputs = tf.concat([
                util.flat(obs_input, self.env.observation_space.shape),
                util.flat(act_input, self.env.action_space.shape)], axis=1)

        theta_output = tf.identity(util.apply_ff(inputs, hid_sizes=[]),
                                   name="theta_output")
        return theta_output

    def build_phi_network(self, old_obs_input, new_obs_input):
        old_o = util.flat(old_obs_input, self.env.observation_space.shape)
        new_o = util.flat(new_obs_input, self.env.observation_space.shape)

        with tf.variable_scope("ff", reuse=tf.AUTO_REUSE):
            old_shaping_output = tf.identity(
                util.apply_ff(old_o, hid_sizes=[self._units] * 2),
                name="old_shaping_output")
            new_shaping_output = tf.identity(
                util.apply_ff(new_o, hid_sizes=[self._units] * 2),
                name="new_shaping_output")
        return old_shaping_output, new_shaping_output


class BasicRewardNet(RewardNet):
    """
    An unshaped reward net. Meant to match the reward network trained for
    the original AIRL pendulum experiments. Right now it has a linear function
    approximator for the theta network, not sure if this is what I want.
    """

    def __init__(self, env, *, state_only=False, **kwargs):
        """
        Params:
          env (gym.Env or str): The environment that we are predicting reward
            for.
          state_only (bool): If True, then ignore the action when predicting
            and training the reward network phi.
        """
        self.state_only = state_only
        super().__init__(env, **kwargs)

    def build_theta_network(self, obs_input, act_input):
        if self.state_only:
            inputs = util.flat(obs_input)
        else:
            inputs = tf.concat([
                util.flat(obs_input, self.env.observation_space.shape),
                util.flat(act_input, self.env.action_space.shape)], axis=1)

        theta_output = tf.identity(util.apply_ff(inputs, hid_sizes=[]),
                                   name="theta_output")
        return theta_output

    @property
    def reward_output_train(self):
        """
        The training reward is the same as the test reward since there is
        no shaping.
        """
        return self.reward_output_test
