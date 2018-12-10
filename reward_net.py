import tensorflow as tf
import tf.keras as keras
from keras import models, layers

# Immediate TODO: Use tf.layers instead of keras.


class RewardNet():

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
        self.old_obs_ph, self.action_ph, self.new_obs_ph = phs

        old_obs_input = keras.Input(tensor=obs_ph, name="old_obs_input")
        action_input = keras.Input(tensor=action_ph, name="action_input")
        new_obs_input = keras.Input(tensor=new_obs_ph, name="new_obs_input")

        self.theta_network, reward = self._build_theta_network(
                old_obs_input, action_input)
        self.phi_network, old_shaping, new_shaping = self._build_phi_network(
                old_obs_input, new_obs_input)
        self.f_network = self._build_f_network(old_obs_input, action_input,
                new_obs_input, reward, old_shaping, new_shaping)

    def _build_placeholders(self):
        """
        Returns old_obs, action, new_obs
        """
        obs_size = TODO(self.env.obs_space)
        action_size = TODO(self.env.action_space)

        # TODO: What about discrete spaces? Should I configure one_hot
        # inputs instead?
        old_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, obs_size])
        action_ph = tf.placeholder(dtype=tf.float32, shape=[None, action_size])
        new_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, obs_size])
        return old_obs, action_ph, new_obs_ph


    def _build_theta_network(self, obs_input, action_input):
        """
        Returns theta_network, a keras Model that predicts reward whose
        input is a state-action pair.

        Although AIRL doesn't individually optimize this subnetwork during
        training, it ends up being our reward approximator at test time.
        """
        inputs = [obs_inputs, action_input]

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
                inputs=[obs_input, action_inputs],
                outputs=reward)

        return model, reward


    def _build_phi_network(self, old_obs_input, new_obs_input):
        """
        Return phi_network, the reward shaping network (serves to disentangle
        dynamics from reward).
        """
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


    def _build_f_network(self, old_obs_input, action_input,
            new_obs_input, reward, old_shaping, new_shaping):
        """
        Combine the theta and phi networks to generate the reward approximator
        used by the AIRL discriminator. (This network is called f(s, a, s') in
        the AIRL paper).
        """

        shaped_reward = reward + self.discount_factor * new_shaping \
                - old_shaping
        model = tf.keras.Model(name="f",
                inputs=[old_obs_input, action_input, new_obs_input],
                outputs=shaped_reward,
                )
        return model
