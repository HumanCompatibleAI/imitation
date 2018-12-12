import tensorflow as tf
from abc import ABC, abstractmethod


class RewardNet(ABC):

    def __init__(self, env, state_only=True, discount_factor=0.9):
        """
        Reward network for Adversarial IRL. Contains two networks, the
        state(-action) reward parameterized by theta, and the state reward
        shaper parameterized by phi.

        Params:
          env (gym.Env): The environment that we are predicting reward for.
            Maybe we could pass in the observation space instead.
          state_only (bool): If True, then ignore the action when predicting
            and training rewards.
          discount_factor (float): A number in the range [0, 1].
        """

        self.env = env
        self.state_only = state_only
        self.discount_factor = discount_factor

        phs = self._build_placeholders()
        self.old_obs_ph, self.act_ph, self.new_obs_ph = phs

        self.reward_output = self.build_theta_network(
                self.old_obs_ph, self.act_ph)

        old_shaping_output, new_shaping_output = self.build_phi_network(
                self.old_obs_ph, self.new_obs_ph)

        self.shaped_reward_output = (self.reward_output +
                self.discount_factor * new_shaping_output - old_shaping_output)


    def _build_placeholders(self):
        """
        Returns old_obs, action, new_obs
        """
        o_shape = (None,) + self.env.observation_space.shape
        a_shape = (None,) + self.env.action_space.shape

        old_obs_ph = tf.placeholder(name="old_obs_ph",
                dtype=tf.float32, shape=o_shape)
        new_obs_ph = tf.placeholder(name="new_obs_ph",
                dtype=tf.float32, shape=o_shape)
        act_ph = tf.placeholder(name="act_ph",
                dtype=tf.float32, shape=a_shape)

        return old_obs, act_ph, new_obs_ph


    @abstractmethod
    def build_theta_network(self, obs_input, act_input):
        """
        Returns theta_network, a keras Model that predicts reward whose
        input is a state-action pair.

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
        shaping_output_old (Tensor) -- A reward shaping prediction for each of
          the old inputs.
        shaping_output_new (Tensor) -- A reward shaping prediction for each of
          the new outputs.
        """
        pass


class BasicRewardNet(RewardNet):

    def build_theta_network(self, obs_input, act_input):
        inputs = [obs_inputs, act_input]

        if not self.state_only:
            x = layers.concatenate(inputs)
        else:
            # Ignore actions, while allowing them to be inputted into the
            # network. (Actually, this setup *requires* that the unused actions
            # are inputted.)
            x = obs_inputs

        # TODO:
        # Just insert dense layers for now, we can somehow have arbitrary
        # layers later.
        x = layers.Dense(64, activation='relu')(x)
        reward = layers.Dense(1)(x)

        model = tf.keras.Model(name="theta",
                inputs=[obs_input, act_inputs],
                outputs=reward)

        return model, reward


    def build_phi_network(self, old_obs_input, new_obs_input):
        # TODO: Parameter instead of magic # for the '64' part. (This seems less
        # urgent than getting something up and running.)
        shared_layers = [
                layers.Dense(64, activation='relu'),
                layers.Dense(1),
                ]

        def apply_shared(x):
            for layer in shared_layers:
                x = layer(x)
            return x

        inputs = [old_obs_input, new_obs_input]
        outputs = old_shaping, new_shaping = [apply_share(x) for x in inputs]
        model = tf.keras.Model(name="phi", inputs=inputs, outputs=outputs)

        return model, old_shaping, new_shaping
