from abc import ABC, abstractmethod

import tensorflow as tf
import yairl.util as util


class DiscrimNet(ABC):
    """Base class for discriminator. Flexible enough to be used in different IRL methods."""
    def __init__(self):
        self._disc_loss = self.build_disc_loss()
        self._policy_train_reward = self.build_policy_train_reward()
        self._policy_test_reward = self.build_policy_test_reward()

    @property
    def disc_loss(self):
        return self._disc_loss

    @property
    def policy_train_reward(self):
        return self._policy_train_reward

    @property
    def policy_test_reward(self):
        return self._policy_test_reward

    @abstractmethod
    def build_policy_train_reward(self):
        pass

    @abstractmethod
    def build_policy_test_reward(self):
        pass

    @abstractmethod
    def build_disc_loss(self):
        # Holds the label of every state-action pair that the discriminator
        # is being trained on. Use 0.0 for expert policy. Use 1.0 for generated
        # policy.
        self.labels_ph = tf.placeholder(shape=(None,), dtype=tf.int32,
                                        name="discrim_labels")

    @abstractmethod
    def build_summaries(self):
        pass


class DiscrimNetAIRL(DiscrimNet):
    """The discriminator to use for AIRL. This discriminator uses a RewardNet."""
    def __init__(self, reward_net):
        self.reward_net = reward_net
        super().__init__()
        self.old_obs_ph = self.reward_net.old_obs_ph
        self.act_ph = self.reward_net.act_ph
        self.new_obs_ph = self.reward_net.new_obs_ph

        tf.logging.info("Using AIRL")

    def build_summaries(self):
        self.reward_net.build_summaries()

    def build_disc_loss(self):
        super().build_disc_loss()
        # Holds the generator-policy log action probabilities of every
        # state-action pair that the discriminator is being trained on.
        self.log_policy_act_prob_ph = tf.placeholder(shape=(None,),
                                                     dtype=tf.float32, name="log_ro_act_prob_ph")

        # Construct discriminator logits by stacking predicted rewards
        # and log action probabilities.
        self._presoftmax_disc_logits = tf.stack(
            [self.reward_net.reward_output_train,
             self.log_policy_act_prob_ph],
            axis=1, name="presoftmax_discriminator_logits")  # (None, 2)

        # Construct discriminator loss.
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels_ph,
            logits=self._presoftmax_disc_logits,
            name="discrim_loss"
        )

    def build_policy_test_reward(self):
        super().build_policy_test_reward()
        return self.reward_net.reward_output_test

    def build_policy_train_reward(self):
        """
        Sets self._policy_train_reward_fn, the reward function to use when
        running a policy optimizer (e.g. PPO).
        """
        # Construct generator reward.
        self._log_softmax_logits = tf.nn.log_softmax(
            self._presoftmax_disc_logits)
        self._log_D, self._log_D_compl = tf.split(
            self._log_softmax_logits, [1, 1], axis=1)
        return self._log_D - self._log_D_compl

class DiscrimNetGAIL(DiscrimNet):
    def __init__(self, env):
        self.env = util.maybe_load_env(env)

        phs = util.build_placeholders(self.env, True)
        self.old_obs_ph, self.act_ph, self.new_obs_ph = phs

        self.log_policy_act_prob_ph = tf.placeholder(shape=(None,),
                                                     dtype=tf.float32, name="log_ro_act_prob_ph")

        with tf.variable_scope("discrim_network"):
            self._discrim_logits = self.build_discrm_network(
                    self.old_obs_ph, self.act_ph)

        super().__init__()

        tf.logging.info("using GAIL")

    def build_discrm_network(self, obs_input, act_input):
        inputs = tf.concat([
            util.flat(obs_input, self.env.observation_space.shape),
            util.flat(act_input, self.env.action_space.shape)], axis=1)

        discrim_logits = util.apply_ff(inputs, hid_sizes=[])

        return discrim_logits

    def build_policy_train_reward(self):
        super().build_policy_train_reward()
        train_reward = -tf.log_sigmoid(self._discrim_logits)

        return train_reward

    def build_policy_test_reward(self):
        super().build_policy_test_reward()
        return self.build_policy_train_reward()

    def build_disc_loss(self):
        super().build_disc_loss()

        disc_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self._discrim_logits,
            labels=tf.cast(self.labels_ph, tf.float32)
        )

        return disc_loss

    def build_summaries(self):
        super().build_summaries()
        tf.summary.histogram("discrim_logits", self._discrim_logits)
