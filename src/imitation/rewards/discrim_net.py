import collections
import os
import pickle
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Sequence, Tuple

import gym
import numpy as np
import tensorflow as tf

from imitation.rewards import reward_net
from imitation.util import networks, serialize


class DiscrimNet(serialize.Serializable, ABC):
    """Abstract base class for discriminator, used in AIRL and GAIL."""

    def __init__(self):
        self._sess = tf.get_default_session()

        # Dict from names to scalar Tensors that are logged by
        # `AdversarialTrainer.train_disc_step`.
        self._train_stats = collections.OrderedDict()  # type: Dict[str, tf.Tensor]

        # Build necessary placeholders, then construct rest of the graph.
        # _labels_gen_is_one_ph holds the label of every state-action pair that the
        # discriminator is being trained on. Use 0.0 for expert policy. Use 1.0 for
        # generated policy.
        self._labels_gen_is_one_ph = tf.placeholder(
            shape=(None,), dtype=tf.int32, name="discrim_labels_gen_is_one"
        )
        # This placeholder holds the generator-policy log action probabilities,
        # $\log \pi(a \mid s)$, of each state-action pair. This includes both
        # actions taken by the generator *and* those by the expert (we can
        # ask our policy for action probabilities even if we don't take them).
        # TODO(adam): this is only used by AIRL; sad we have to always include it
        self._log_policy_act_prob_ph = tf.placeholder(
            shape=(None,), dtype=tf.float32, name="log_ro_act_prob_ph"
        )

        self._disc_loss = None  # type: tf.Tensor
        self._policy_train_reward = None  # type: tf.Tensor
        self._policy_test_reward = None  # type: tf.Tensor
        self._disc_logits_gen_is_high = None  # type: tf.Tensor

        self.build_graph()

        assert self._disc_loss is not None
        assert self._policy_train_reward is not None
        assert self._policy_test_reward is not None
        assert self._disc_logits_gen_is_high is not None

        self._add_default_train_stats()
        tf.summary.histogram("disc_logits", self.disc_logits_gen_is_high)

    @property
    @abstractmethod
    def obs_ph(self):
        """The previous observation placeholder."""

    @property
    @abstractmethod
    def act_ph(self):
        """The action placeholder."""

    @property
    @abstractmethod
    def next_obs_ph(self):
        """The new observation placeholder."""

    @property
    @abstractmethod
    def done_ph(self):
        """The episode termination placeholder."""

    @property
    def labels_gen_is_one_ph(self):
        """The expert (0.0) or generated (1.0) labels placeholder."""
        return self._labels_gen_is_one_ph

    @property
    def log_policy_act_prob_ph(self):
        """The log-probability of policy actions placeholder."""
        return self._log_policy_act_prob_ph

    @property
    def disc_logits_gen_is_high(self) -> tf.Tensor:
        """The discriminator's logits for each state-action sample.

        A high value corresponds to predicting generator, and a low value corresponds to
        predicting expert.
        """
        return self._disc_logits_gen_is_high

    @property
    def disc_loss(self) -> tf.Tensor:
        return self._disc_loss  # pytype: disable=attribute-error

    @property
    def policy_train_reward(self) -> tf.Tensor:
        return self._policy_train_reward  # pytype: disable=attribute-error

    @property
    def policy_test_reward(self) -> tf.Tensor:
        return self._policy_test_reward  # pytype: disable=attribute-error

    @property
    def train_stats(self) -> Dict[str, tf.Tensor]:
        """A feed dictionary of scalar Tensors to be logged during training."""
        return self._train_stats

    @abstractmethod
    def build_graph(self):
        """Builds forward prop graph, reward, loss, and summary ops.

        Gets called once during construction. Should create the following
        attributes:

        - `self._policy_train_reward`: reward used during imitation learning,
          should be a rank-1 Tensor.
        - `self._policy_test_reward`: reward used during testing, should be rank-1
          Tensor. This is useful for AIRL, where it can be used to drop shaping
          terms, but for GAIL it will just be an alias for _policy_train_reward.
        - `self._disc_loss`: discriminator loss to be optimised.

        FIXME(sam): remove train reward/test reward distinction.
        """

    def _add_default_train_stats(self) -> None:
        """Updates `self._train_stats` with several default scalar logging Tensors."""
        bin_is_generated_pred = self.disc_logits_gen_is_high > 0
        bin_is_generated_true = self.labels_gen_is_one_ph > 0
        bin_is_expert_true = tf.logical_not(bin_is_generated_true)
        int_is_generated_pred = tf.cast(bin_is_generated_pred, tf.int32)
        int_is_generated_true = tf.cast(bin_is_generated_true, tf.int32)
        n_generated = tf.reduce_sum(int_is_generated_true)
        n_labels = tf.size(self.labels_gen_is_one_ph)
        n_expert = n_labels - n_generated
        pct_expert = tf.cast(n_expert, tf.float32) / tf.cast(n_labels, tf.float32)
        n_expert_pred = tf.size(bin_is_generated_pred) - tf.reduce_sum(
            int_is_generated_pred
        )
        pct_expert_pred = tf.cast(n_expert_pred, tf.float32) / tf.cast(
            n_labels, tf.float32
        )
        correct_vec = tf.equal(bin_is_generated_pred, bin_is_generated_true)
        acc = tf.reduce_mean(tf.cast(correct_vec, tf.float32))

        _n_pred_expert = tf.reduce_sum(
            tf.cast(tf.logical_and(bin_is_expert_true, correct_vec), tf.float32)
        )
        _n_expert_or_1 = tf.cast(tf.maximum(1, n_expert), tf.float32)
        expert_acc = _n_pred_expert / _n_expert_or_1

        _n_pred_gen = tf.reduce_sum(
            tf.cast(tf.logical_and(bin_is_generated_true, correct_vec), tf.float32)
        )
        _n_gen_or_1 = tf.cast(tf.maximum(1, n_generated), tf.float32)
        generated_acc = _n_pred_gen / _n_gen_or_1

        label_dist = tf.distributions.Bernoulli(logits=self.disc_logits_gen_is_high)
        entropy = tf.reduce_mean(label_dist.entropy())

        self._train_stats.update(
            [
                ("disc_xent_loss", tf.reduce_mean(self._disc_loss)),
                # accuracy, as well as accuracy on *just* expert examples and *just*
                # generated examples
                ("disc_acc", acc),
                ("disc_acc_expert", expert_acc),
                ("disc_acc_gen", generated_acc),
                # entropy of the predicted label distribution, averaged equally across
                # both classes (if this drops then disc is very good or has given up)
                ("disc_entropy", entropy),
                # true number of expert demos and predicted number of expert demos
                ("disc_proportion_expert_true", pct_expert),
                ("disc_proportion_expert_pred", pct_expert_pred),
                ("n_expert", n_expert),
                ("n_generated", n_generated),
            ]
        )

    def reward_train(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        *,
        gen_log_prob_fn: Callable[..., np.ndarray],
    ) -> np.ndarray:
        """Vectorized reward for training an imitation learning algorithm.

        Args:
            obs: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            act: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_obs: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            dones: Whether the episode has terminated. Its shape is `(batch_size,)`.
            gen_log_prob_fn: The generator policy's action probabilities function.
                A Callable such that
                `log_act_prob_fn(observations=obs, actions=act, lopg=True)`
                returns `log_act_prob`, the generator's log action probabilities.
                `log_act_prob[i]` is equal to the generator's log probability of
                choosing `act[i]` given `obs[i]`.
        Returns:
            The rewards. Its shape is `(batch_size,)`.
        """
        log_act_prob = gen_log_prob_fn(
            observation=obs, actions=act, logp=True
        ).flatten()

        n_gen = len(obs)
        assert obs.shape == next_obs.shape
        assert len(act) == n_gen
        assert log_act_prob.shape == (n_gen,)

        fd = {
            self.obs_ph: obs,
            self.act_ph: act,
            self.next_obs_ph: next_obs,
            self.done_ph: dones,
            self.labels_gen_is_one_ph: np.ones(n_gen),
            self.log_policy_act_prob_ph: log_act_prob,
        }
        rew = self._sess.run(self.policy_train_reward, feed_dict=fd)
        assert rew.shape == (n_gen,)
        return rew

    def reward_test(
        self, obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray, dones: np.ndarray,
    ) -> np.ndarray:
        """Vectorized reward for training an expert during transfer learning.

        Args:
            obs: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            act: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_obs: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            dones: Whether the episode has terminated. Its shape is `(batch_size,)`.
        Returns:
            The rewards. Its shape is `(batch_size,)`.
        """
        fd = {
            self.obs_ph: obs,
            self.act_ph: act,
            self.next_obs_ph: next_obs,
            self.done_ph: dones,
        }
        rew = self._sess.run(self.policy_test_reward, feed_dict=fd)
        assert rew.shape == (len(obs),)
        return rew


class DiscrimNetAIRL(DiscrimNet):
    r"""The AIRL discriminator for a given RewardNet.

    The AIRL discriminator is of the form
    .. math:: D_{\theta}(s,a) = \frac{\exp(f_{\theta}(s,a)}{\exp(f_{\theta}(s,a) + \pi(a \mid s)}

    where :math:`f_{\theta}` is `self.reward_net`.
    """  # noqa: E501

    def __init__(self, reward_net: reward_net.RewardNet, entropy_weight: float = 1.0):
        """Builds a DiscrimNetAIRL.

        Args:
            reward_net: A RewardNet, used as $f_{\theta}$ in the discriminator.
            entropy_weight: The coefficient for the entropy regularization term.
                To match the AIRL derivation, it should be 1.0.
                However, empirically a lower value sometimes work better.
        """
        self.reward_net = reward_net
        self.entropy_weight = entropy_weight
        super().__init__()
        tf.logging.info("Using AIRL")

    @property
    def obs_ph(self):
        return self.reward_net.obs_ph

    @property
    def act_ph(self):
        return self.reward_net.act_ph

    @property
    def next_obs_ph(self):
        return self.reward_net.next_obs_ph

    @property
    def done_ph(self):
        return self.reward_net.done_ph

    def build_graph(self):
        self._disc_logits_gen_is_high = (
            self.log_policy_act_prob_ph - self.reward_net.reward_output_train
        )

        # Construct discriminator loss.
        self._disc_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(self.labels_gen_is_one_ph, tf.float32),
            logits=self._disc_logits_gen_is_high,
            name="disc_loss",
        )  # (None,)

        # Construct generator reward:
        # \[\hat{r}(s,a) = \log(D_{\theta}(s,a)) - \log(1 - D_{\theta}(s,a)).\]
        # This simplifies to:
        # \[\hat{r}(s,a) = f_{\theta}(s,a) - \log \pi(a \mid s).\]
        # This is just an entropy-regularized objective.

        ent_bonus = -self.entropy_weight * self.log_policy_act_prob_ph
        self._policy_train_reward = self.reward_net.reward_output_train + ent_bonus
        self._policy_test_reward = self.reward_net.reward_output_test

        self.reward_net.build_summaries()

    def _save(self, directory):
        os.makedirs(directory, exist_ok=True)
        params = {
            "entropy_weight": self.entropy_weight,
            "reward_net_cls": type(self.reward_net),
        }
        with open(os.path.join(directory, "args"), "wb") as f:
            pickle.dump(params, f)
        return self.reward_net.save(os.path.join(directory, "reward_net"))

    @classmethod
    def _load(cls, directory):
        with open(os.path.join(directory, "args"), "rb") as f:
            params = pickle.load(f)
        reward_net_cls = params.pop("reward_net_cls")
        reward_net = reward_net_cls.load(os.path.join(directory, "reward_net"))
        return cls(reward_net=reward_net, **params)


DiscrimNetBuilder = Callable[
    [Sequence[tf.Tensor]], Tuple[tf.Tensor, networks.LayersDict]
]
"""Type alias for function that builds a discriminator network.

Takes an observation and action tensor and produces a tuple containing
(1) a list of used TF layers, and (2) output logits.
"""


class DiscrimNetGAIL(DiscrimNet, serialize.LayersSerializable):
    """The discriminator to use for GAIL."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        build_discrim_net: DiscrimNetBuilder = networks.build_and_apply_mlp,
        build_discrim_net_kwargs: Optional[dict] = None,
        scale: bool = False,
    ):
        """Construct discriminator network.

        Args:
          observation_space: observation space for this environment.
          action_space: action space for this environment:
          build_discrim_net: a callable that takes an observation input tensor
            and action input tensor as input, then computes the logits
            necessary to feed to GAIL. When called, the function should return
            *both* a `LayersDict` containing all the layers used in
            construction of the discriminator network, and a `tf.Tensor`
            representing the desired discriminator logits.
          build_discrim_net_kwargs: optional extra keyword arguments for
            `build_discrim_net()`.
          scale: should inputs be rescaled according to declared observation
            space bounds?
        """
        # for serialisation
        args = dict(locals())

        # things we'll need in .build_graph()
        self._observation_space = observation_space
        self._action_space = action_space
        self._scale = scale
        self._build_discrim_net = build_discrim_net
        self._build_discrim_net_kwargs = build_discrim_net_kwargs or {}

        self._disc_mlp = None
        # Builds graph via call to `self.build_graph()`.
        DiscrimNet.__init__(self)
        assert self._disc_mlp is not None

        # records args for un-pickling as well as newly-created model
        serialize.LayersSerializable.__init__(**args, layers=self._disc_mlp)

        tf.logging.info("using GAIL")

    @property
    def obs_ph(self):
        return self._obs_ph

    @property
    def act_ph(self):
        return self._act_ph

    @property
    def next_obs_ph(self):
        return self._next_obs_ph

    @property
    def done_ph(self):
        return self._done_ph

    def build_graph(self):
        phs, inps = networks.build_inputs(
            self._observation_space, self._action_space, scale=self._scale
        )
        self._obs_ph, self._act_ph, self._next_obs_ph, self._done_ph = phs
        self.obs_input, self.act_input, _, self.done_input = inps

        with tf.variable_scope("discrim_network"):
            self._disc_logits_gen_is_high, self._disc_mlp = self._build_discrim_net(
                [self.obs_input, self.act_input], **self._build_discrim_net_kwargs
            )
        self._policy_test_reward = self._policy_train_reward = -tf.log_sigmoid(
            self._disc_logits_gen_is_high
        )

        self._disc_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self._disc_logits_gen_is_high,
            labels=tf.cast(self.labels_gen_is_one_ph, tf.float32),
        )
