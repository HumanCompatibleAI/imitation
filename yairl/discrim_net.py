from abc import ABC, abstractmethod

import tensorflow as tf


class DiscrimNet(ABC):
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
    def __init__(self, reward_net):
        self.reward_net = reward_net
        super().__init__()
        self.old_obs_ph = self.reward_net.old_obs_ph
        self.act_ph = self.reward_net.act_ph
        self.new_obs_ph = self.reward_net.new_obs_ph

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
