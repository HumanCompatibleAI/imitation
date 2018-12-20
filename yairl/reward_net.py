from abc import ABC, abstractmethod

import tensorflow as tf

import yairl.util as util


class RewardNet(ABC):

    def __init__(self, env, discount_factor=0.9):
        """
        Reward network for Adversarial IRL. Contains two networks, the
        state(-action) reward parameterized by theta, and the state reward
        shaper parameterized by phi.

        This network is session-less -- we assume that the caller will
        initialize the network's variables.

        Params:
          env (gym.Env or str): The environment that we are predicting reward
            for.
          discount_factor (float): A number in the range [0, 1].
        """

        self.env = util.maybe_load_env(env)
        self.discount_factor = discount_factor

        phs = self._build_placeholders()
        self.old_obs_ph, self.act_ph, self.new_obs_ph = phs

        with tf.variable_scope("theta_network"):
            self.reward_output = self.build_theta_network(
                    self.old_obs_ph, self.act_ph)

        with tf.variable_scope("phi_network"):
            res = self.build_phi_network(self.old_obs_ph, self.new_obs_ph)
            self.old_shaping_output, self.new_shaping_output = res

        with tf.variable_scope("f_network"):
            self.shaped_reward_output = (self.reward_output +
                    self.discount_factor * self.new_shaping_output
                    - self.old_shaping_output)
        # TODO: assert that all the outputs above have shape (None,).

    def _build_placeholders(self):
        """
        Returns old_obs_ph, act_ph, new_obs_ph
        """
        o_shape = (None,) + self.env.observation_space.shape
        a_shape = (None,) + self.env.action_space.shape

        old_obs_ph = tf.placeholder(name="old_obs_ph",
                dtype=tf.float32, shape=o_shape)
        new_obs_ph = tf.placeholder(name="new_obs_ph",
                dtype=tf.float32, shape=o_shape)
        act_ph = tf.placeholder(name="act_ph",
                dtype=tf.float32, shape=a_shape)

        return old_obs_ph, act_ph, new_obs_ph

    @abstractmethod
    def build_theta_network(self, obs_input, act_input):
        """
        Build the reward network.

        Although AIRL doesn't individually optimize this subnetwork during
        training, it ends up being our reward approximator at test time.

        Params:
        obs_input (Tensor) -- The observation input. Its shape is
          `((None,) + self.env.observation_space.shape)`.
        act_input (Tensor) -- The action input. Its shape is
          `((None,) + self.env.action_space.shape)`. The None dimension is
          expected to be the same as None dimension from `obs_input`.

        Return:
        reward_output (Tensor) -- A reward prediction for each of the
          inputs. The shape is `(None,)`.
        """
        pass

    # TODO: Add an option to ignore phi network.
    @abstractmethod
    def build_phi_network(self, old_obs_input, new_obs_input):
        """
        Build the reward shaping network (serves to disentangle dynamics from
        reward).

        XXX: We could potentially make it easier on the subclasser by requiring
          only one input. ie build_phi_network(obs_input). Later in
          _build_f_network, I could stack Tensors of old and new observations,
          pass them simulatenously through the network, and then unstack the
          outputs.

        Params:
        old_obs_input (Tensor): The old observations (corresponding to the
          state at which the current action is made). The shape of this
          Tensor should be `((None,) + self.env.observation_space.shape)`.
        new_obs_input (Tensor): The new observations (corresponding to the
          state that we transition to after this state-action pair.

        Return:
        old_shaping_output (Tensor) -- A reward shaping prediction for each of
          the old observation inputs. Has shape `(None,)` (batch size).
        new_shaping_output (Tensor) -- A reward shaping prediction for each of
          the new observation inputs. Has shape `(None,)` (batch size).
        """
        pass


class BasicRewardNet(RewardNet):
    """
    A reward network with default settings. Meant for prototyping.

    This network flattens inputs. So probably don't use an RGB observation
    space.
    """

    def __init__(self, env, ignore_action=False, **kwargs):
        """
        Params:
          env (gym.Env or str): The environment that we are predicting reward
            for.
          ignore_action (bool): If True, then ignore the action when predicting
            and training rewards.
          discount_factor (float): A number in the range [0, 1].
        """
        self.ignore_action = ignore_action
        super().__init__(env, **kwargs)

    def _apply_ff(self, inputs):
        """
        Apply the a default feed forward network on the inputs. The Dense
        layers are auto_reused. (Recall that build_*_network() called inside
        a var scope named *_network.)
        """
        # TODO: Parameter instead of magic # for the '64' part. (This seems less
        # urgent than getting something up and running.)
        # XXX: Seems like xavier is default?
        # https://stackoverflow.com/q/37350131/1091722
        xavier = tf.contrib.layers.xavier_initializer
        x = tf.layers.dense(inputs, 64, activation='relu',
                kernel_initializer=xavier(), name="dense1")
        x = tf.layers.dense(x, 1, kernel_initializer=xavier(),
                name="dense2")
        return tf.squeeze(x, axis=1)

    def build_theta_network(self, obs_input, act_input):
        if self.ignore_action:
            inputs = flat(obs_input)
        else:
            inputs = tf.concat([
                _flat(obs_input, self.env.observation_space.shape),
                _flat(act_input, self.env.action_space.shape)], axis=1)

        reward_output = tf.identity(self._apply_ff(inputs),
                "reward_output")
        return reward_output

    def build_phi_network(self, old_obs_input, new_obs_input):
        old_o = _flat(old_obs_input, self.env.observation_space.shape)
        new_o = _flat(new_obs_input, self.env.observation_space.shape)

        with tf.variable_scope("ff", reuse=tf.AUTO_REUSE):
            old_shaping_output = tf.identity(self._apply_ff(old_o),
                    name="old_shaping_output")
            new_shaping_output = tf.identity(self._apply_ff(new_o),
                    name="new_shaping_output")
        return old_shaping_output, new_shaping_output


def _flat(tensor, space_shape):
    ndim = len(space_shape)
    if ndim== 0:
        return tf.reshape(tensor, [-1, 1])
    elif ndim == 1:
        return tf.reshape(tensor, [-1, space_shape[0]])
    else:
        # TODO: Take the product(space_shape) and use that as the final
        # dimension. In fact, product could encompass all the previous
        # cases.
        raise NotImplementedError
