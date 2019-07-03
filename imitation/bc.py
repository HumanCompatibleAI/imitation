"""Behavioural cloning. Trains policy by applying learning to a supplied
dataset of (observation, action) pairs."""

import tensorflow as tf
from tqdm.autonotebook import tqdm, trange

from stable_baselines.common.dataset import Dataset

from imitation.util import rollout, FeedForward32Policy


class BCTrainer(object):
  """Simple Behavioural cloning (BC). Recovers only a policy."""

  def __init__(self,
               env,
               *,
               expert_trainers,
               policy_class=FeedForward32Policy,
               n_expert_timesteps=4000,
               batch_size=32):
    self.env = env
    self.policy_class = policy_class
    self.expert_trainers = expert_trainers
    self.batch_size = batch_size
    expert_obs, expert_acts, expert_nobs = rollout.generate_multiple(
        expert_trainers, self.env, n_expert_timesteps)
    self.expert_dataset = Dataset(
        {
            "obs": expert_obs,
            "act": expert_acts,
        }, shuffle=True)
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    with self.graph.as_default():
      self._build_tf_graph()
      self.sess.run(tf.global_variables_initializer())

  def train(self, *, n_epochs=100):
    """Train with supervised learning for some number of epochs. Here an
        'epoch' is just a complete pass through the dataset."""
    epoch_iter = trange(n_epochs, desc='BC epoch')
    for epoch_num in epoch_iter:
      total_batches = self.expert_dataset.n_samples // self.batch_size
      batch_iter = self.expert_dataset.iterate_once(self.batch_size)
      tq_iter = tqdm(
          batch_iter, total=total_batches, desc='pol step', leave=False)
      loss_ewma = None
      for batch_dict in tq_iter:
        feed_dict = {
            self._true_acts_ph: batch_dict['act'],
            self.policy.obs_ph: batch_dict['obs'],
        }
        _, loss = self.sess.run(
            [self._train_op, self._log_loss], feed_dict=feed_dict)
        tq_iter.set_postfix(loss='% 3.4f' % loss)
        if loss_ewma is None:
          loss_ewma = loss
        else:
          loss_ewma = 0.9 * loss_ewma + 0.1 * loss
      epoch_iter.set_postfix(loss_ewma='% 3.4f' % loss_ewma)

  def test_policy(self, *, n_episodes=10):
    """Test current imitation policy on environment & give mean episode
    return under true reward function."""
    total_reward = rollout.total_reward(
        self.policy, self.env, n_episodes=n_episodes)
    mean_reward = total_reward / float(n_episodes)
    return mean_reward

  def _build_tf_graph(self):
    with tf.name_scope('bc_supervised_loss'):
      # self._obs_ph = tf.placeholder(
      #     tf.float32,
      #     shape=(None, ) + self.env.observation_space.shape,
      #     name='obs')
      self.policy = self.policy_class(
          self.sess,
          self.env.observation_space,
          self.env.action_space,
          # why does a "policy" class need to take any of these
          # arguments? surely n_steps etc. should be totally independent
          # of the policy
          n_batch=None,
          n_env=1,
          # ???
          n_steps=1000)
      self._true_acts_ph = self.policy.pdtype.sample_placeholder(
          [None], name='ref_acts_ph')
      self._log_loss = tf.reduce_mean(
          self.policy.proba_distribution.neglogp(self._true_acts_ph))
      opt = tf.train.AdamOptimizer()
      self._train_op = opt.minimize(self._log_loss)
